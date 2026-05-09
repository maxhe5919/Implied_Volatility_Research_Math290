from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
from config import MONEYNESS_NODES, logger

SYNTHETIC_DTE_DAYS = np.array([7, 14, 21, 30, 45], dtype=float)
SYNTHETIC_TAU_YEARS = SYNTHETIC_DTE_DAYS / 365.25
LOG_MONEYNESS_NODES = np.log(MONEYNESS_NODES)

WINDOW_MINUTES = 60
N_DTE = len(SYNTHETIC_DTE_DAYS)
N_MONEY = len(MONEYNESS_NODES)

SVI_PARAM_RANGES = {
    "a":     (0.0003, 0.003),
    "b":     (0.05,  0.15),
    "rho":   (-0.85, -0.10),
    "m":     (-0.10, 0.10),
    "sigma": (0.05,  0.25),
}
SVI_PARAM_NAMES = ["a", "b", "rho", "m", "sigma"]

OU_KAPPA = 5.0
OU_SIGMA_PCT = 0.05
DT_TRADING = 1.0 / 390.0

MAX_SLICE_REJECTIONS = 200
MAX_OU_CLIPS_PER_STEP = 50
MIN_ACCEPT_RATE_WARN = 0.05

PERTURBATION_CLASSES = ["P0_clean", "P1", "P2", "P3", "P4"]

@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SVIParams":
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])


def raw_svi(k: np.ndarray, p: SVIParams) -> np.ndarray:
    z = k - p.m
    return p.a + p.b * (p.rho * z + np.sqrt(z * z + p.sigma * p.sigma))


def raw_svi_derivs(k: np.ndarray, p: SVIParams):
    z = k - p.m
    denom = np.sqrt(z * z + p.sigma * p.sigma)
    w_prime = p.b * (p.rho + z / denom)
    w_double = p.b * p.sigma * p.sigma / (denom ** 3)
    return w_prime, w_double


def butterfly_g(k: np.ndarray, p: SVIParams) -> np.ndarray:
    w = raw_svi(k, p)
    w_prime, w_double = raw_svi_derivs(k, p)
    w_safe = np.maximum(w, 1e-10)

    term1 = (k * w_prime / (2.0 * w_safe)) ** 2
    term2 = (w_prime ** 2) / 4.0 * (1.0 / w_safe + 0.25)
    term3 = w_double / 2.0

    return 1.0 - term1 - term2 + term3

def is_butterfly_free(p: SVIParams, k_grid: np.ndarray) -> bool:
    g = butterfly_g(k_grid, p)
    return bool(np.all(g >= -1e-8))


def is_calendar_free(params_by_slice, k_grid: np.ndarray) -> bool:
    n_slices = len(params_by_slice)
    if n_slices < 2:
        return True
    w_curr = raw_svi(k_grid, params_by_slice[0])
    for j in range(1, n_slices):
        w_next = raw_svi(k_grid, params_by_slice[j])
        if np.any(w_next < w_curr - 1e-8):
            return False
        w_curr = w_next
    return True


def is_arbitrage_free(params_by_slice, k_grid: np.ndarray) -> bool:
    for p in params_by_slice:
        if not is_butterfly_free(p, k_grid):
            return False
    return is_calendar_free(params_by_slice, k_grid)

def _sample_param_box(rng: np.random.Generator) -> SVIParams:
    return SVIParams(
        a=rng.uniform(*SVI_PARAM_RANGES["a"]),
        b=rng.uniform(*SVI_PARAM_RANGES["b"]),
        rho=rng.uniform(*SVI_PARAM_RANGES["rho"]),
        m=rng.uniform(*SVI_PARAM_RANGES["m"]),
        sigma=rng.uniform(*SVI_PARAM_RANGES["sigma"]),
    )


def sample_svi_slice(rng: np.random.Generator,
                     k_grid: np.ndarray) -> Optional[SVIParams]:
    for _ in range(MAX_SLICE_REJECTIONS):
        p = _sample_param_box(rng)
        if is_butterfly_free(p, k_grid):
            return p
    return None


def sample_svi_surface(rng: np.random.Generator,
                       k_grid: np.ndarray,
                       n_slices: int = N_DTE,
                       tau_years: np.ndarray = SYNTHETIC_TAU_YEARS
                       ) -> Optional[list[SVIParams]]:
    if n_slices != len(tau_years):
        raise ValueError(f"n_slices ({n_slices}) != len(tau_years) ({len(tau_years)})")

    a_lo, a_hi = SVI_PARAM_RANGES["a"]
    b_lo, b_hi = SVI_PARAM_RANGES["b"]
    tau_ratios = tau_years / tau_years[0]
    alpha = 0.5

    for _ in range(MAX_SLICE_REJECTIONS):
        rho = rng.uniform(*SVI_PARAM_RANGES["rho"])
        m = rng.uniform(*SVI_PARAM_RANGES["m"])
        sigma = rng.uniform(*SVI_PARAM_RANGES["sigma"])

        a_base_max = a_hi / tau_ratios[-1]
        a_base = rng.uniform(a_lo, a_base_max)
        b_base_max = b_hi / (tau_ratios[-1] ** alpha)
        b_base = rng.uniform(b_lo, b_base_max)
        slices = []
        all_ok = True
        for j in range(n_slices):
            a_j = a_base * tau_ratios[j]
            b_j = b_base * (tau_ratios[j] ** alpha)
            p = SVIParams(a=a_j, b=b_j, rho=rho, m=m, sigma=sigma)
            if not is_butterfly_free(p, k_grid):
                all_ok = False
                break
            slices.append(p)
        if not all_ok:
            continue
        if not is_calendar_free(slices, k_grid):
            continue

        return slices

    return None

def _ou_step(theta_prev: np.ndarray,
             theta_bar: np.ndarray,
             range_widths: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    drift = OU_KAPPA * (theta_bar - theta_prev) * DT_TRADING
    diffusion = OU_SIGMA_PCT * range_widths * np.sqrt(DT_TRADING) * rng.standard_normal(len(theta_prev))
    return theta_prev + drift + diffusion


def _clip_to_box(theta: np.ndarray) -> np.ndarray:
    out = theta.copy()
    for i, name in enumerate(SVI_PARAM_NAMES):
        lo, hi = SVI_PARAM_RANGES[name]
        out[i] = np.clip(out[i], lo, hi)
    return out


def generate_synthetic_window(rng: np.random.Generator,
                              k_grid: np.ndarray = LOG_MONEYNESS_NODES,
                              tau_years: np.ndarray = SYNTHETIC_TAU_YEARS,
                              n_minutes: int = WINDOW_MINUTES
                              ) -> Optional[np.ndarray]:

    params_per_slice = sample_svi_surface(rng, k_grid, n_slices=len(tau_years))
    if params_per_slice is None:
        return None

    theta_bar = np.array([p.to_array() for p in params_per_slice])
    theta_curr = theta_bar.copy()

    range_widths = np.array(
        [SVI_PARAM_RANGES[name][1] - SVI_PARAM_RANGES[name][0]
         for name in SVI_PARAM_NAMES]
    )

    iv_window = np.zeros((n_minutes, len(k_grid), len(tau_years)), dtype=float)

    for t in range(n_minutes):
        for j in range(len(tau_years)):
            new_theta = _ou_step(theta_curr[j], theta_bar[j], range_widths, rng)
            new_theta = _clip_to_box(new_theta)

            for _ in range(MAX_OU_CLIPS_PER_STEP):
                p_new = SVIParams.from_array(new_theta)
                if not is_butterfly_free(p_new, k_grid):
                    new_theta = 0.5 * (new_theta + theta_bar[j])
                    new_theta = _clip_to_box(new_theta)
                    continue
                if j > 0:
                    w_prev = raw_svi(k_grid, SVIParams.from_array(theta_curr[j - 1]))
                    w_new = raw_svi(k_grid, p_new)
                    if np.any(w_new < w_prev - 1e-8):
                        new_theta = 0.5 * (new_theta + theta_bar[j])
                        new_theta = _clip_to_box(new_theta)
                        continue
                break
            else:
                new_theta = theta_curr[j]

            theta_curr[j] = new_theta

        for j, tau in enumerate(tau_years):
            p = SVIParams.from_array(theta_curr[j])
            w = raw_svi(k_grid, p)
            w = np.maximum(w, 1e-10)
            iv_window[t, :, j] = np.sqrt(w / tau)

    return iv_window

def inject_perturbation(window: np.ndarray,
                        ptype: str,
                        magnitude: float,
                        rng: np.random.Generator):

    T, M, D = window.shape
    out = window.copy()
    mask = np.zeros_like(window, dtype=bool)

    sigma_noise = float(np.std(window))
    if ptype == "P0_clean":
        return out, mask

    elif ptype == "P1":
        t = rng.integers(0, T)
        i = rng.integers(0, M)
        j = rng.integers(0, D)
        sign = rng.choice([-1.0, 1.0])
        out[t, i, j] += sign * magnitude * sigma_noise
        mask[t, i, j] = True

    elif ptype == "P2":
        t_start = rng.integers(0, max(1, T - 10))
        t_end = min(T, t_start + 10)
        u = rng.standard_normal(M); u /= np.linalg.norm(u)
        v = rng.standard_normal(D); v /= np.linalg.norm(v)
        ramp = np.linspace(0, 1, t_end - t_start)
        sign = rng.choice([-1.0, 1.0])
        for s, t in enumerate(range(t_start, t_end)):
            out[t] += sign * magnitude * sigma_noise * ramp[s] * np.outer(u, v)

    elif ptype == "P3":
        t = rng.integers(0, T)
        j = rng.integers(0, D)
        bump_width = rng.integers(3, 5)
        i_start = rng.integers(0, M - bump_width)
        x = np.arange(bump_width)
        center = (bump_width - 1) / 2.0
        bump_shape = np.exp(-0.5 * ((x - center) / (bump_width / 4.0)) ** 2)
        sign = rng.choice([-1.0, 1.0])
        out[t, i_start:i_start + bump_width, j] += sign * magnitude * sigma_noise * bump_shape
        mask[t, i_start:i_start + bump_width, j] = True

    elif ptype == "P4":
        if magnitude is not None:
            pass
        noise_scale_pct = 0.015
        ref_iv = float(np.median(window))
        noise_std = noise_scale_pct * ref_iv

        coords = np.stack(np.meshgrid(np.arange(M), np.arange(D), indexing="ij"), axis=-1
                          ).reshape(-1, 2).astype(float)
        diff = coords[:, None, :] - coords[None, :, :]
        diff[..., 0] /= 2.0
        diff[..., 1] /= 1.0
        cov = np.exp(-0.5 * np.sum(diff ** 2, axis=-1))
        try:
            L_chol = np.linalg.cholesky(cov + 1e-6 * np.eye(M * D))
        except np.linalg.LinAlgError:
            L_chol = np.linalg.cholesky(cov + 1e-3 * np.eye(M * D))

        for t in range(T):
            z = rng.standard_normal(M * D)
            noise = (L_chol @ z).reshape(M, D)
            out[t] += noise_std * noise

    else:
        raise ValueError(f"Unknown perturbation class: {ptype}")

    return out, mask


def generate_dataset(n_train: int = 5000,
                     n_val: int = 1000,
                     n_test: int = 2000,
                     magnitudes=(0.5, 1.0, 2.0, 3.0),
                     seed: int = 42,
                     output_path: Optional[Path] = None):
    rng = np.random.default_rng(seed)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    out = {}
    n_failures = 0

    for split_name, n_samples in splits.items():
        logger.info(f"Generating {split_name}: {n_samples} windows")
        windows_clean = np.empty((n_samples, WINDOW_MINUTES, N_MONEY, N_DTE), dtype=np.float32)
        windows_dirty = np.empty_like(windows_clean)
        masks = np.zeros((n_samples, WINDOW_MINUTES, N_MONEY, N_DTE), dtype=bool)
        ptypes = np.empty(n_samples, dtype=object)
        mags = np.empty(n_samples, dtype=np.float32)

        i = 0
        attempts = 0
        while i < n_samples:
            attempts += 1
            clean = generate_synthetic_window(rng)
            if clean is None:
                n_failures += 1
                if n_failures % 50 == 0:
                    logger.warning(
                        f"Window generation failures: {n_failures} "
                        f"(accept rate {i / max(attempts, 1):.2%})"
                    )
                continue

            ptype = rng.choice(PERTURBATION_CLASSES, p=[0.25, 0.1875, 0.1875, 0.1875, 0.1875])
            if ptype in ("P0_clean", "P4"):
                mag = 1.0
            else:
                mag = rng.choice(magnitudes)
            dirty, mask = inject_perturbation(clean, ptype, mag, rng)

            windows_clean[i] = clean
            windows_dirty[i] = dirty
            masks[i] = mask
            ptypes[i] = ptype
            mags[i] = mag

            i += 1
            if i % 500 == 0:
                logger.info(f"  {split_name}: {i}/{n_samples}")

        accept_rate = n_samples / max(attempts, 1)
        if accept_rate < MIN_ACCEPT_RATE_WARN:
            logger.warning(
                f"{split_name} acceptance rate {accept_rate:.2%} below "
                f"{MIN_ACCEPT_RATE_WARN:.0%}. Parameter ranges may be too loose."
            )

        out[split_name] = {
            "windows_clean": windows_clean,
            "windows_dirty": windows_dirty,
            "masks": masks,
            "ptypes": ptypes,
            "magnitudes": mags,
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **{f"{split}_{k}": v
               for split, d in out.items()
               for k, v in d.items()}
        )
        logger.info(f"Dataset saved to {output_path}")

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = SVIParams(a=0.02, b=0.15, rho=-0.6, m=0.0, sigma=0.2)
    g = butterfly_g(LOG_MONEYNESS_NODES, p)
    print(f"Sample slice butterfly g(k) min/max: {g.min():.4f} / {g.max():.4f}")
    print(f"Butterfly free? {is_butterfly_free(p, LOG_MONEYNESS_NODES)}")

    win = generate_synthetic_window(rng)
    if win is None:
        print("FAIL: window generation returned None")
    else:
        print(f"Generated window shape: {win.shape}")
        print(f"IV range: [{win.min():.4f}, {win.max():.4f}]")
        print(f"IV mean: {win.mean():.4f}")

    for pt in PERTURBATION_CLASSES:
        dirty, mask = inject_perturbation(win, pt, magnitude=2.0, rng=rng)
        n_anomaly = int(mask.sum())
        delta = float(np.abs(dirty - win).max())
        print(f"  {pt}: anomaly cells = {n_anomaly:5d}, max delta = {delta:.4f}")