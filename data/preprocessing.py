import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import norm
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
from config import (DATA_DIR, MAX_DTE_TERMS, MONEYNESS_NODES, RISK_FREE_RATE, logger)

GAP_THRESHOLD_MIN = 5
MIN_DTE_DAYS = 0
def vectorized_implied_volatility(prices, S, K, T_years, r, flags,
                                  tol=1e-5, max_iter=100):

    sigma = np.full_like(prices, 0.30, dtype=float)
    is_call = flags == 'C'

    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
        d2 = d1 - sigma * np.sqrt(T_years)

        price_est = np.where(
            is_call,
            S * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2),
            K * np.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1),
        )
        vega = np.maximum(S * norm.pdf(d1) * np.sqrt(T_years), 1e-6)
        step = (price_est - prices) / vega
        sigma = np.clip(sigma - step, 0.001, 5.0)

        if np.nanmax(np.abs(price_est - prices)) < tol:
            break

    return np.where((sigma > 0.001) & (sigma < 4.99), sigma, np.nan)


def load_align_and_filter(symbol: str, date_str: str):
    logger.info(f"Loading {symbol} on {date_str}")
    opt_file = os.path.join(DATA_DIR, "options", symbol, f"{symbol}_snapshot_{date_str}.csv.gz")
    stk_file = os.path.join(DATA_DIR, "stocks",  symbol, f"{symbol}_ticks_{date_str}.csv.gz")

    df_opt = pd.read_csv(opt_file)
    df_stk = pd.read_csv(stk_file)

    df_opt['TimestampLocal'] = pd.to_datetime(
        df_opt['TimestampLocal'], format='ISO8601', errors='coerce'
    ).dt.tz_localize(None)
    df_stk['TimestampLocal'] = pd.to_datetime(
        df_stk['TimestampLocal'], format='ISO8601', errors='coerce'
    ).dt.tz_localize(None)

    df_opt = df_opt.dropna(subset=['TimestampLocal'])
    df_stk = df_stk.dropna(subset=['TimestampLocal'])
    opt_min, opt_max = df_opt['TimestampLocal'].min(), df_opt['TimestampLocal'].max()
    df_stk = df_stk[(df_stk['TimestampLocal'] >= opt_min) &
                    (df_stk['TimestampLocal'] <= opt_max)]

    df_opt['TimeKey'] = df_opt['TimestampLocal'].dt.floor('1min')
    df_opt = df_opt.drop_duplicates(subset=['TimeKey', 'OptionSymbol'], keep='last')
    df_opt['TimestampLocal'] = df_opt['TimeKey']

    df_merged = pd.merge_asof(
        left=df_opt.sort_values('TimestampLocal'),
        right=df_stk[['TimestampLocal', 'LastPrice']].sort_values('TimestampLocal'),
        on='TimestampLocal',
        direction='backward',
    ).dropna(subset=['LastPrice'])

    df_merged['Right'] = df_merged['OptionSymbol'].str[-9]
    df_merged['Strike'] = df_merged['OptionSymbol'].str[-8:].astype(float) / 1000.0
    df_merged['MidPrice'] = (df_merged['Bid'] + df_merged['Ask']) / 2.0
    df_merged['Expiration'] = (
        pd.to_datetime(df_merged['OptionSymbol'].str[-15:-9], format='%y%m%d')
        + pd.Timedelta(hours=16)
    )
    df_merged['T_Years'] = (
        df_merged['Expiration'] - df_merged['TimestampLocal']
    ).dt.total_seconds() / (365.25 * 24 * 3600)
    df_merged = df_merged[df_merged['T_Years'] > 0.001]

    trading_date = pd.to_datetime(date_str)
    df_merged['Static_DTE'] = (
        df_merged['Expiration'].dt.normalize() - trading_date
    ).dt.days
    n_before = (df_merged['Static_DTE'] < MIN_DTE_DAYS).sum()
    if n_before > 0:
        logger.info(
            f"Dropping {n_before} option rows with DTE < {MIN_DTE_DAYS} "
            "(numerical IV inversion fragility + pin risk)."
        )
    df_merged = df_merged[df_merged['Static_DTE'] >= MIN_DTE_DAYS]

    available_dtes = np.sort(df_merged['Static_DTE'].unique())
    dynamic_dte_nodes = available_dtes[:MAX_DTE_TERMS]
    logger.info(f"DTE nodes: {dynamic_dte_nodes}")

    if len(dynamic_dte_nodes) < MAX_DTE_TERMS:
        logger.warning(
            f"Only {len(dynamic_dte_nodes)} DTE nodes available after "
            f"MIN_DTE_DAYS={MIN_DTE_DAYS} filter; expected {MAX_DTE_TERMS}. "
            "Downstream feature matrix will have correspondingly fewer columns."
        )

    df_merged = df_merged[df_merged['Static_DTE'].isin(dynamic_dte_nodes)]
    df_merged['Moneyness'] = df_merged['Strike'] / df_merged['LastPrice']

    df_full_quotes = df_merged.copy()
    otm_puts  = df_merged[(df_merged['Right'] == 'P') & (df_merged['Moneyness'] <= 1.0)]
    otm_calls = df_merged[(df_merged['Right'] == 'C') & (df_merged['Moneyness'] >  1.0)]
    clean = pd.concat([otm_puts, otm_calls])

    clean['Custom_IV'] = vectorized_implied_volatility(
        clean['MidPrice'].values, clean['LastPrice'].values,
        clean['Strike'].values, clean['T_Years'].values,
        RISK_FREE_RATE, clean['Right'].values,
    )

    return clean.dropna(subset=['Custom_IV']), df_full_quotes, dynamic_dte_nodes

def detect_gaps(timestamps: pd.DatetimeIndex, threshold_min: int = GAP_THRESHOLD_MIN):
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) < 2:
        return np.zeros(len(ts), dtype=bool), np.zeros(len(ts))

    deltas = np.diff(ts.values).astype('timedelta64[s]').astype(float) / 60.0
    deltas = np.append(deltas, 0.0)
    gap_after = deltas > threshold_min
    return gap_after, deltas


def gaps_in_window(timestamps: pd.DatetimeIndex,
                   window_minutes: int = 60,
                   threshold_min: int = GAP_THRESHOLD_MIN):
    ts = pd.DatetimeIndex(timestamps)
    T = len(ts)
    contaminated = np.zeros(T, dtype=bool)
    if T < 2:
        return contaminated
    deltas_min = np.diff(ts.values).astype('timedelta64[s]').astype(float) / 60.0
    for t in range(T):
        window_start = ts[t] - pd.Timedelta(minutes=window_minutes)
        j = ts.searchsorted(window_start, side='left')
        if j >= t:
            continue
        if np.any(deltas_min[j:t] > threshold_min):
            contaminated[t] = True

    return contaminated

def build_feature_matrix(df: pd.DataFrame, dynamic_dte_nodes: np.ndarray):
    logger.info("Building causal feature matrix.")

    timestamps = np.sort(df['TimestampLocal'].unique())

    num_features = len(MONEYNESS_NODES) * len(dynamic_dte_nodes)
    M = np.full((len(timestamps), num_features), np.nan)

    grid_x, grid_y = np.meshgrid(MONEYNESS_NODES, dynamic_dte_nodes)
    valid_rows = []
    valid_ts = []

    for ts in timestamps:
        time_slice = df[df['TimestampLocal'] == ts]
        if len(time_slice) < 4:
            continue

        points = time_slice[['Moneyness', 'Static_DTE']].values
        values = time_slice['Custom_IV'].values

        try:
            grid_lin  = griddata(points, values, (grid_x, grid_y), method='linear')
            grid_near = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z = np.where(np.isnan(grid_lin), grid_near, grid_lin)
            valid_rows.append(grid_z.flatten())
            valid_ts.append(ts)
        except Exception as e:
            logger.warning(f"Interpolation failed at {ts}: {e}")
            continue

    if not valid_rows:
        raise RuntimeError("No timestamps yielded a valid IV grid.")

    M = np.array(valid_rows)
    timestamps = pd.DatetimeIndex(valid_ts)
    gap_after, deltas = detect_gaps(timestamps, threshold_min=GAP_THRESHOLD_MIN)
    gap_flags = np.zeros(len(timestamps), dtype=bool)
    if len(timestamps) > 1:
        gap_flags[1:] = deltas[:-1] > GAP_THRESHOLD_MIN

    M_df = pd.DataFrame(M)
    M_df_filled = M_df.ffill()
    M_filled = M_df_filled.values
    for i in np.where(gap_flags)[0]:
        M_filled[i] = M[i]

    n_total_nan = np.isnan(M_filled).sum()
    if n_total_nan > 0:
        logger.warning(
            f"{n_total_nan} NaN cells remain after gap-aware ffill "
            f"({100 * n_total_nan / M_filled.size:.2f}% of matrix). "
            "Downstream rolling driver must handle these."
        )

    n_real_gaps = int(gap_flags.sum())
    logger.info(
        f"Feature matrix shape: {M_filled.shape}; "
        f"{n_real_gaps} rows flagged with preceding gap > {GAP_THRESHOLD_MIN} min."
    )

    return M_filled, timestamps, gap_flags