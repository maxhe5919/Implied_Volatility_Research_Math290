from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PERTURBATION_CLASSES = ("P0_clean", "P1", "P2", "P3", "P4")

class SyntheticIVDataset(Dataset):
    def __init__(self,
                 npz_path: str | Path,
                 split: str,
                 filter_ptype: Optional[str] = None,
                 filter_magnitude: Optional[float] = None,
                 dtype: torch.dtype = torch.float32):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.npz_path = Path(npz_path)
        self.split = split
        self.dtype = dtype
        with np.load(self.npz_path, allow_pickle=True) as f:
            X_dirty = f[f"{split}_windows_dirty"]
            L_clean = f[f"{split}_windows_clean"]
            mask = f[f"{split}_masks"]
            ptypes = f[f"{split}_ptypes"]
            mags = f[f"{split}_magnitudes"]

        keep_idx = np.arange(len(X_dirty))
        if filter_ptype is not None:
            if filter_ptype not in PERTURBATION_CLASSES:
                raise ValueError(
                    f"filter_ptype must be one of {PERTURBATION_CLASSES}, "
                    f"got {filter_ptype!r}"
                )
            keep_idx = keep_idx[ptypes[keep_idx] == filter_ptype]
        if filter_magnitude is not None:
            keep_idx = keep_idx[
                np.isclose(mags[keep_idx], filter_magnitude, atol=1e-6)
            ]

        if len(keep_idx) == 0:
            raise ValueError(
                f"No items match filter (ptype={filter_ptype!r}, "
                f"magnitude={filter_magnitude}) in split {split!r}"
            )
        N, T, M, D = X_dirty.shape
        self.T = T
        self.M = M
        self.D = D
        self.F = M * D

        self.X_dirty = X_dirty[keep_idx].reshape(-1, T, M * D).astype(np.float32)
        self.L_clean = L_clean[keep_idx].reshape(-1, T, M * D).astype(np.float32)

        self.S_anom = (self.X_dirty - self.L_clean).copy()
        ptype_arr = ptypes[keep_idx].astype(str)
        null_mask = np.isin(ptype_arr, ("P0_clean", "P4"))
        if null_mask.any():
            self.S_anom[null_mask] = 0.0
            self.L_clean[null_mask] = self.X_dirty[null_mask]
            n_null = int(null_mask.sum())
            n_total = len(ptype_arr)
            print(f"  [{split}] relabeled {n_null}/{n_total} null-case windows (P0_clean + P4): L<-X_dirty, S<-0")
        self.mask = mask[keep_idx].reshape(-1, T, M * D).astype(np.bool_)
        self.ptypes = ptypes[keep_idx].astype(str)
        self.magnitudes = mags[keep_idx].astype(np.float32)
        self.original_indices = keep_idx

        for arr_name, arr in (("X_dirty", self.X_dirty),
                              ("L_clean", self.L_clean)):
            if not np.isfinite(arr).all():
                bad_per_window = (~np.isfinite(arr)).reshape(arr.shape[0], -1).any(axis=1)
                bad_idx = np.where(bad_per_window)[0]
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                raise ValueError(
                    f"{split}.{arr_name} contains non-finite values: "
                    f"{n_nan} NaN, {n_inf} Inf across {len(bad_idx)} window(s). "
                    f"First bad indices: {bad_idx[:10].tolist()}. "
                    f"This is a generator bug -- regenerate the dataset, "
                    f"or filter these windows before training."
                )

    def __len__(self) -> int:
        return len(self.X_dirty)

    def __getitem__(self, idx: int):
        X = torch.from_numpy(self.X_dirty[idx]).to(self.dtype)
        L = torch.from_numpy(self.L_clean[idx]).to(self.dtype)
        S = torch.from_numpy(self.S_anom[idx]).to(self.dtype)
        return X, L, S

    def get_metadata(self, idx: int) -> dict:
        return {
            "ptype": str(self.ptypes[idx]),
            "magnitude": float(self.magnitudes[idx]),
            "original_index": int(self.original_indices[idx]),
        }

    def get_mask(self, idx: int) -> np.ndarray:
        return self.mask[idx]


