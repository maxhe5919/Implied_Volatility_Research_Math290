"""
The Classical Rank Constrained RPCA (RC-RPCA) via rank-r truncated IALM.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .base import Decomposer


def soft_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)


def rank_r_threshold(M: np.ndarray, tau: float, r: int) -> np.ndarray:
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    U_r = U[:, :r]
    sigma_r = sigma[:r]
    Vt_r = Vt[:r, :]
    sigma_r_thresh = soft_threshold(sigma_r, tau)
    return (U_r * sigma_r_thresh) @ Vt_r


@dataclass
class RCRPCAConfig:
    rank: int = 3
    K: int | None = None
    tol: float = 1e-7
    max_iter: int = 500
    rho: float = 1.5
    mu_scale: float = 1.25
    lam_scale: float | None = None

    def __post_init__(self):
        if self.K is not None and self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


class RCRPCA(Decomposer):
    def __init__(self, config: RCRPCAConfig | None = None):
        self.config = config if config is not None else RCRPCAConfig()
        self.rank = self.config.rank
        self._last_n_iter: int | None = None
        self._last_residual: float | None = None
        self._last_converged: bool | None = None

    def fit_window(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (T, F), got shape {X.shape}")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN. Preprocessing should fill or skip these before decomposition.")

        cfg = self.config
        T, F = X.shape
        norm_X = np.linalg.norm(X, 'fro')

        if norm_X == 0.0:
            self._last_n_iter = 0
            self._last_residual = 0.0
            self._last_converged = True
            return np.zeros_like(X), np.zeros_like(X)

        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)

        spectral_norm_X = np.linalg.norm(X, 2)
        mu = cfg.mu_scale / max(spectral_norm_X, 1e-12)
        lam_base = cfg.lam_scale if cfg.lam_scale is not None else 1.0
        lam = lam_base / np.sqrt(max(T, F))

        if cfg.K is not None:
            n_iter_target = cfg.K
            check_convergence = False
        else:
            n_iter_target = cfg.max_iter
            check_convergence = True

        residual = np.inf
        converged = False

        for k in range(n_iter_target):
            L = rank_r_threshold(X - S + Y / mu, tau=1.0 / mu, r=cfg.rank)
            S = soft_threshold(X - L + Y / mu, tau=lam / mu)
            Z = X - L - S
            Y = Y + mu * Z
            mu = mu * cfg.rho
            residual = np.linalg.norm(Z, 'fro') / norm_X
            if check_convergence and residual < cfg.tol:
                converged = True
                break

        self._last_n_iter = k + 1
        self._last_residual = float(residual)
        self._last_converged = converged or (cfg.K is not None and k + 1 == cfg.K)

        return L, S

    @property
    def last_n_iter(self) -> int | None:
        return self._last_n_iter

    @property
    def last_residual(self) -> float | None:
        return self._last_residual

    @property
    def last_converged(self) -> bool | None:
        return self._last_converged

    def __repr__(self) -> str:
        cfg = self.config
        if cfg.K is not None:
            mode = f"K={cfg.K}"
        else:
            mode = f"converge(tol={cfg.tol:.0e}, max_iter={cfg.max_iter})"
        return f"RCRPCA(rank={cfg.rank}, mode={mode})"


def rcrpca_fixed_K(K: int = 8, rank: int = 3) -> RCRPCA:
    return RCRPCA(RCRPCAConfig(rank=rank, K=K))


def rcrpca_converged(tol: float = 1e-7, max_iter: int = 500, rank: int = 3) -> RCRPCA:
    return RCRPCA(RCRPCAConfig(rank=rank, K=None, tol=tol, max_iter=max_iter))
