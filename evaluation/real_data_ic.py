from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unrolled_rc_rpca import URCRPCA
from data.preprocessing import (load_align_and_filter, build_feature_matrix, MONEYNESS_NODES,
)

REAL_EVAL_MAX_DTE_TERMS = 5

def load_with_front_dtes(symbol: str, date_str: str, max_terms: int = REAL_EVAL_MAX_DTE_TERMS):
    df_iv, df_full_quotes, dte_nodes = load_align_and_filter(symbol, date_str)
    return df_iv, df_full_quotes, dte_nodes

def batched_rolling_decomposition(model: URCRPCA,
                                   M_mat: np.ndarray,
                                   window: int,
                                   device: torch.device,
                                   batch_size: int = 64) -> np.ndarray:
    T, F = M_mat.shape
    S_pred_causal = np.full((T, F), np.nan, dtype=np.float32)
    valid_window_indices = []
    valid_t_targets = []
    for t in range(window - 1, T):
        X_window = M_mat[t - window + 1 : t + 1]
        if not np.isnan(X_window).any():
            valid_window_indices.append(X_window)
            valid_t_targets.append(t)

    if not valid_window_indices:
        return S_pred_causal

    X_all = np.stack(valid_window_indices, axis=0)
    X_torch = torch.from_numpy(X_all).float()
    n_total = X_all.shape[0]

    model.eval()
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            chunk = X_torch[i:i + batch_size].to(device)
            _, S_batch = model(chunk)
            S_last = S_batch[:, -1, :].cpu().numpy()
            for j, t in enumerate(valid_t_targets[i:i + batch_size]):
                S_pred_causal[t] = S_last[j]

    return S_pred_causal


def compute_signals_and_pnl(df_iv: pd.DataFrame,
                            df_quotes: pd.DataFrame,
                            M_matrix: np.ndarray,
                            S_pred: np.ndarray,
                            dte_nodes: np.ndarray,
                            timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    atm_money_idx = int(np.argmin(np.abs(MONEYNESS_NODES - 1.0)))
    atm_dte_idx = 0
    M = len(MONEYNESS_NODES)
    flat_atm_idx = atm_dte_idx * M + atm_money_idx
    eligible_trade_dtes = dte_nodes[dte_nodes >= 10]
    if len(eligible_trade_dtes) == 0:
        trade_dte = float(dte_nodes[-1])
    else:
        trade_dte = float(eligible_trade_dtes[0])
    underlying = (df_iv.drop_duplicates(subset=['TimestampLocal'])
                       [['TimestampLocal', 'LastPrice']]
                       .set_index('TimestampLocal')
                       .sort_index())
    underlying['LogRet'] = np.log(underlying['LastPrice'] / underlying['LastPrice'].shift(1))
    quotes_grouped = dict(tuple(df_quotes.groupby('TimestampLocal')))

    n_no_strike = 0
    n_no_quotes = 0
    n_no_exit = 0
    n_no_rv = 0

    for i, t in enumerate(timestamps):
        if i < 60:
            continue
        if np.isnan(S_pred[i]).any():
            continue

        target_horizons = [5, 10, 15, 30, 60]
        exit_slices = {}
        missing_exit = False

        for m in target_horizons:
            t_exit = t + pd.Timedelta(minutes=m)
            if t_exit not in quotes_grouped:
                missing_exit = True
                break
            exit_slices[f"{m}m"] = quotes_grouped[t_exit]

        if missing_exit:
            n_no_exit += 1
            continue

        slice_t = quotes_grouped[t]
        cands = slice_t[slice_t['Static_DTE'] == trade_dte]
        if cands.empty:
            n_no_strike += 1
            continue
        idx_atm = int(np.argmin(np.abs(cands['Moneyness'].values - 1.0)))
        atm_strike = cands.iloc[idx_atm]['Strike']
        call_t = slice_t[(slice_t['Strike'] == atm_strike) & (slice_t['Right'] == 'C') & (slice_t['Static_DTE'] == trade_dte)]
        put_t = slice_t[(slice_t['Strike'] == atm_strike) & (slice_t['Right'] == 'P') & (slice_t['Static_DTE'] == trade_dte)]
        if call_t.empty or put_t.empty:
            n_no_quotes += 1
            continue

        c_bid_t, c_ask_t, c_mid_t = call_t.iloc[0][['Bid', 'Ask', 'MidPrice']]
        p_bid_t, p_ask_t, p_mid_t = put_t.iloc[0][['Bid', 'Ask', 'MidPrice']]
        rv_window = underlying.loc[t - pd.Timedelta(minutes=30):t, 'LogRet'].dropna()
        if len(rv_window) < 15:
            n_no_rv += 1
            continue
        trail_rv = float(rv_window.std() * np.sqrt(252 * 390))

        record = {
            'Timestamp': t,
            'Signal_URC': float(S_pred[i, flat_atm_idx]),
            'B1_Raw_IV': float(M_matrix[i, flat_atm_idx]),
            'B3_Trail_RV': trail_rv,
            'B4_IV_RV_Spread': float(M_matrix[i, flat_atm_idx]) - trail_rv,
        }

        for h_label, slice_exit in exit_slices.items():
            call_ex = slice_exit[(slice_exit['Strike'] == atm_strike) &
                                 (slice_exit['Right'] == 'C') &
                                 (slice_exit['Static_DTE'] == trade_dte)]
            put_ex = slice_exit[(slice_exit['Strike'] == atm_strike) &
                                (slice_exit['Right'] == 'P') &
                                (slice_exit['Static_DTE'] == trade_dte)]
            if call_ex.empty or put_ex.empty:
                record[f'PnL_{h_label}_Mid'] = np.nan
                record[f'PnL_{h_label}_Half'] = np.nan
                record[f'PnL_{h_label}_Full'] = np.nan
                continue

            c_bid_x, c_ask_x, c_mid_x = call_ex.iloc[0][['Bid', 'Ask', 'MidPrice']]
            p_bid_x, p_ask_x, p_mid_x = put_ex.iloc[0][['Bid', 'Ask', 'MidPrice']]

            entry_mid = c_mid_t + p_mid_t
            exit_mid = c_mid_x + p_mid_x
            record[f'PnL_{h_label}_Mid'] = (exit_mid - entry_mid) / entry_mid

            entry_half = entry_mid + 0.25 * ((c_ask_t - c_bid_t) + (p_ask_t - p_bid_t))
            exit_half = exit_mid - 0.25 * ((c_ask_x - c_bid_x) + (p_ask_x - p_bid_x))
            record[f'PnL_{h_label}_Half'] = (exit_half - entry_half) / entry_half

            entry_full = c_ask_t + p_ask_t
            exit_full = c_bid_x + p_bid_x
            record[f'PnL_{h_label}_Full'] = (exit_full - entry_full) / entry_full

        records.append(record)

    if not records:
        print(f"    drop reasons: no_strike={n_no_strike}, no_quotes={n_no_quotes}, "
              f"no_exit={n_no_exit}, no_rv={n_no_rv}", flush=True)

    return pd.DataFrame(records)

def add_time_of_day_baseline(df: pd.DataFrame, target_col: str) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=float)
    df = df.copy()
    df['_minute_of_day'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
    df['_date'] = df['Timestamp'].dt.normalize()
    by_minute = df.groupby('_minute_of_day')

    for minute_of_day, g in by_minute:
        per_date_means = g.groupby('_date')[target_col].mean()
        sum_total = per_date_means.sum()
        n_dates = len(per_date_means)
        if n_dates < 2:
            continue
        for idx in g.index:
            this_date = df.loc[idx, '_date']
            this_date_mean = per_date_means.get(this_date, 0.0)
            out[df.index.get_loc(idx)] = (sum_total - this_date_mean) / (n_dates - 1)

    return out

def bootstrap_ic_day_block(symbol_df: pd.DataFrame,
                           signals: list[str],
                           targets: list[str],
                           n_boot: int = 10000,
                           seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dates = symbol_df['Timestamp'].dt.normalize().unique()
    n_dates = len(dates)
    if n_dates < 2:
        return {(sig, tgt): {"mean": np.nan, "lower": np.nan, "upper": np.nan}
                for sig in signals for tgt in targets}
    per_date_arrays: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {
        (sig, tgt): [] for sig in signals for tgt in targets
    }
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df_by_date = {d: g for d, g in symbol_df.groupby(symbol_df['Timestamp'].dt.normalize())}

    for d in dates:
        g = df_by_date[d]
        for sig in signals:
            for tgt in targets:
                if sig not in g.columns or tgt not in g.columns:
                    continue
                s = g[sig].values.astype(float)
                t = g[tgt].values.astype(float)
                m = np.isfinite(s) & np.isfinite(t)
                if m.sum() < 2:
                    per_date_arrays[(sig, tgt)].append((np.empty(0), np.empty(0)))
                else:
                    per_date_arrays[(sig, tgt)].append((s[m], t[m]))
    out = {}
    for (sig, tgt), per_date_list in per_date_arrays.items():
        ic_samples = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sample_idx = rng.integers(0, n_dates, size=n_dates)
            xs, ys = [], []
            for k in sample_idx:
                s_arr, t_arr = per_date_list[k]
                if len(s_arr) > 0:
                    xs.append(s_arr)
                    ys.append(t_arr)
            if not xs:
                ic_samples[b] = np.nan
                continue
            x_concat = np.concatenate(xs)
            y_concat = np.concatenate(ys)
            if len(x_concat) < 3:
                ic_samples[b] = np.nan
                continue
            rx = rankdata(x_concat)
            ry = rankdata(y_concat)
            rx_c = rx - rx.mean()
            ry_c = ry - ry.mean()
            denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
            ic_samples[b] = (rx_c * ry_c).sum() / denom if denom > 0 else np.nan

        valid = ic_samples[np.isfinite(ic_samples)]
        if len(valid) == 0:
            out[(sig, tgt)] = {"mean": np.nan, "lower": np.nan, "upper": np.nan}
        else:
            out[(sig, tgt)] = {
                "mean": float(np.mean(valid)),
                "lower": float(np.quantile(valid, 0.025)),
                "upper": float(np.quantile(valid, 0.975)),
            }

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directory with best.pt or final.pt")
    p.add_argument("--symbols", nargs='+', default=["AAPL", "NVDA", "TSLA"], help="Mag-7 single-names with daily expirations")
    p.add_argument("--dates", nargs='+', required=True, help="List of YYYY-MM-DD to evaluate")
    p.add_argument("--n-bootstraps", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--decomp-batch-size", type=int, default=64, help="Windows per forward pass during rolling decomposition")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    K = cfg["K"]
    rank = cfg["rank"]
    T = cfg["T"]
    F_dim = cfg["F_dim"]

    model = URCRPCA(K=K, rank=rank, T=T, F_dim=F_dim).to(device)

    best_path = run_dir / "best.pt"
    final_path = run_dir / "final.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        print(f"Loaded best.pt (val_loss={ckpt.get('val_loss')})", flush=True)
    elif final_path.exists():
        ckpt = torch.load(final_path, map_location=device)
        print(f"WARN: best.pt missing, falling back to final.pt", flush=True)
    else:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: K={K}, rank={rank}, T={T}, F_dim={F_dim}", flush=True)

    signals = ['Signal_URC', 'B1_Raw_IV', 'B2_TimeOfDay', 'B3_Trail_RV', 'B4_IV_RV_Spread']
    horizons = ['5m', '10m', '15m', '30m', '60m']
    costs = ['Mid', 'Half', 'Full']
    targets = [f'PnL_{h}_{c}' for h in horizons for c in costs]

    for symbol in args.symbols:
        print(f"\n{'='*72}\nSymbol: {symbol}\n{'='*72}", flush=True)
        symbol_df_list = []
        symbol_dir = run_dir / "eval" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for date in args.dates:
            print(f"\n[{symbol} {date}] preprocessing...", flush=True)
            try:
                df_iv, df_quotes, dte_nodes = load_with_front_dtes(symbol, date)
                M_mat, ts, gap_flags = build_feature_matrix(df_iv, dte_nodes)
                print(f"  M shape: {M_mat.shape}, DTE nodes: {dte_nodes.tolist()}",
                      flush=True)

                T_data, F_data = M_mat.shape
                if F_data != F_dim:
                    print(f"  WARN: F={F_data} != model F_dim={F_dim}; "
                          f"DTE count differs from training distribution. "
                          f"Skipping day.", flush=True)
                    continue

                print(f"  decomposing (window=60, batched)...", flush=True)
                S_pred = batched_rolling_decomposition(
                    model, M_mat, window=60, device=device,
                    batch_size=args.decomp_batch_size,
                )

                print(f"  computing signals & P&L...", flush=True)
                day_df = compute_signals_and_pnl(
                    df_iv, df_quotes, M_mat, S_pred, dte_nodes, ts
                )
                if day_df.empty:
                    print(f"  no usable rows on this day; skipping.", flush=True)
                    continue
                day_df['Date'] = date
                daily_path = symbol_dir / f"{symbol}_{date}_signals_pnl.csv"
                day_df.to_csv(daily_path, index=False)
                print(f"  saved {len(day_df)} rows to {daily_path.name}",
                      flush=True)
                symbol_df_list.append(day_df)
            except Exception as e:
                print(f"  ERROR on {symbol} {date}: {type(e).__name__}: {e}",
                      flush=True)
                continue

        if not symbol_df_list:
            print(f"\n[{symbol}] no usable days; skipping IC computation.",
                  flush=True)
            continue

        symbol_df = pd.concat(symbol_df_list, ignore_index=True)
        n_obs = len(symbol_df)
        n_days = symbol_df['Timestamp'].dt.normalize().nunique()
        print(f"\n[{symbol}] pooled: {n_obs} observations across {n_days} days",
              flush=True)

        print(f"  computing B2 (leave-one-day-out time-of-day fixed effect)...",
              flush=True)
        symbol_df = symbol_df.reset_index(drop=True)
        symbol_df['B2_TimeOfDay'] = add_time_of_day_baseline(
            symbol_df, target_col='PnL_30m_Mid'
        )
        n_b2_valid = symbol_df['B2_TimeOfDay'].notna().sum()
        print(f"    B2 has {n_b2_valid}/{len(symbol_df)} valid values", flush=True)

        print(f"  bootstrapping IC ({args.n_bootstraps} reps)...", flush=True)
        ic_results = bootstrap_ic_day_block(
            symbol_df, signals, targets,
            n_boot=args.n_bootstraps, seed=args.seed,
        )
        rows = []
        for h in horizons:
            for c in costs:
                target = f'PnL_{h}_{c}'
                print(f"\n--- {symbol} | Horizon: {h} | Cost: {c} ---",
                      flush=True)
                for sig in signals:
                    r = ic_results.get((sig, target),
                                       {"mean": np.nan, "lower": np.nan, "upper": np.nan})
                    print(f"  {sig:>20}: IC = {r['mean']:+.4f}  "
                          f"[95% CI: {r['lower']:+.4f}, {r['upper']:+.4f}]",
                          flush=True)
                    rows.append({
                        'Symbol': symbol,
                        'Horizon': h,
                        'Cost_Regime': c,
                        'Signal': sig,
                        'IC_Mean': r['mean'],
                        'IC_Lower95': r['lower'],
                        'IC_Upper95': r['upper'],
                    })
        stats_df = pd.DataFrame(rows)
        out_path = symbol_dir / f"{symbol}_aggregated_IC_stats.csv"
        stats_df.to_csv(out_path, index=False)
        print(f"\n[{symbol}] saved IC stats to {out_path}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
