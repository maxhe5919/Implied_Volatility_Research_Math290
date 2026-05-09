from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Decomposer(ABC):
    rank: int = 3

    @abstractmethod
    def fit_window(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def fit_rolling(self, X: np.ndarray, window: int, gap_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        T, F = X.shape
        L_out = np.full_like(X, np.nan, dtype=float)
        S_out = np.full_like(X, np.nan, dtype=float)

        for t in range(window, T + 1):
            X_window = X[t - window:t]
            if np.any(np.isnan(X_window)):
                continue
            if gap_mask is not None and gap_mask[t - 1]:
                continue

            L_w, S_w = self.fit_window(X_window)
            L_out[t - 1] = L_w[-1]
            S_out[t - 1] = S_w[-1]

        return L_out, S_out

    def fit_batch(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, T, M, D = X_batch.shape
        L_out = np.empty_like(X_batch)
        S_out = np.empty_like(X_batch)
        for i in range(N):
            X_flat = X_batch[i].reshape(T, M * D)
            L_flat, S_flat = self.fit_window(X_flat)
            L_out[i] = L_flat.reshape(T, M, D)
            S_out[i] = S_flat.reshape(T, M, D)
        return L_out, S_out