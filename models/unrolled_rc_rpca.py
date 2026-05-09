from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from models.rc_rpca import RCRPCA, RCRPCAConfig

def inv_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError(f"inv_softplus requires y > 0, got {y}")
    if y > 20.0:
        return y
    return math.log(math.expm1(y))

def soft_threshold(M: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.sign(M) * F.relu(torch.abs(M) - tau)


class _RankRTruncatedSVT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, tau, r):
        # SVD
        U, sigma, Vh = torch.linalg.svd(M, full_matrices=False)

        # Slice top r
        U_r = U[..., :, :r].contiguous()
        sigma_r = sigma[..., :r].contiguous()
        Vh_r = Vh[..., :r, :].contiguous()

        if tau.ndim >= 2 and tau.shape[-1] == 1 and tau.shape[-2] == 1:
            tau = tau.squeeze(-1)
        active_mask = (sigma_r > tau).to(sigma_r.dtype)
        s_r = F.relu(sigma_r - tau)

        # Reconstruct L
        L = (U_r * s_r.unsqueeze(-2)) @ Vh_r

        ctx.save_for_backward(U_r, s_r, Vh_r, active_mask)
        ctx.r = r
        ctx.tau_was_squeezed = True
        return L

    @staticmethod
    def backward(ctx, grad_L):
        U_r, s_r, Vh_r, active_mask = ctx.saved_tensors
        V_r = Vh_r.transpose(-1, -2)


        UrT_gL = U_r.transpose(-1, -2) @ grad_L
        gL_Vr = grad_L @ V_r
        UrT_gL_Vr = UrT_gL @ V_r

        am_left = active_mask.unsqueeze(-2)
        am_right = active_mask.unsqueeze(-1)

        term1 = U_r @ UrT_gL_Vr @ Vh_r
        gL_Vr_active = gL_Vr * am_left
        full2 = gL_Vr_active @ Vh_r
        proj2 = U_r @ (U_r.transpose(-1, -2) @ full2)
        term2 = full2 - proj2

        UrT_gL_active = UrT_gL * am_right
        full3 = U_r @ UrT_gL_active
        proj3 = (full3 @ V_r) @ Vh_r
        term3 = full3 - proj3

        grad_M = term1 + term2 + term3

        diag_UrT_gL_Vr = torch.diagonal(UrT_gL_Vr, dim1=-2, dim2=-1)
        grad_tau_per_r = -active_mask * diag_UrT_gL_Vr
        grad_tau_summed = grad_tau_per_r.sum(dim=-1, keepdim=True)
        grad_tau = grad_tau_summed.unsqueeze(-1)

        return grad_M, grad_tau, None


def rank_r_truncated_svt(M: torch.Tensor, tau: torch.Tensor, r: int) -> torch.Tensor:
    """ Rank-r truncated singular value soft-thresholding"""
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=M.dtype, device=M.device)
    return _RankRTruncatedSVT.apply(M, tau, r)

class URCRPCALayer(nn.Module):
    def __init__(self, c_mu_init: float, d_L_init: float, d_S_init: float, rank: int = 3):
        super().__init__()
        self.rank = rank
        self.raw_c_mu = nn.Parameter(torch.tensor(inv_softplus(c_mu_init), dtype=torch.float32))
        self.raw_d_L = nn.Parameter(torch.tensor(inv_softplus(d_L_init), dtype=torch.float32))
        self.raw_d_S = nn.Parameter(torch.tensor(inv_softplus(d_S_init), dtype=torch.float32))

    @property
    def c_mu(self) -> torch.Tensor:
        return F.softplus(self.raw_c_mu)

    @property
    def d_L(self) -> torch.Tensor:
        return F.softplus(self.raw_d_L)

    @property
    def d_S(self) -> torch.Tensor:
        return F.softplus(self.raw_d_S)

    def forward(self,
                X: torch.Tensor,
                L_prev: torch.Tensor,
                S_prev: torch.Tensor,
                Y_prev: torch.Tensor,
                X_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mu = self.c_mu / X_norm
        tau_L = X_norm * self.d_L
        tau_S = X_norm * self.d_S

        # low-rank update via rank-r truncated soft-SVT
        L_new = rank_r_truncated_svt(X - S_prev + Y_prev / mu, tau=tau_L, r=self.rank)
        # sparse update
        S_new = soft_threshold(X - L_new + Y_prev / mu, tau=tau_S)
        # dual update
        Y_new = Y_prev + mu * (X - L_new - S_new)

        return L_new, S_new, Y_new


class URCRPCA(nn.Module):
    """ K-layer unrolled rank-constrained RPCA """
    def __init__(self,
                 K: int = 8,
                 rank: int = 3,
                 T: int = 60,
                 F_dim: int = 105,
                 rho: float = 1.5,
                 mu0: float = 1.25,
                 lam_scale: float = 1.0):
        super().__init__()
        self.K = K
        self.rank = rank
        self.T = T
        self.F_dim = F_dim
        lam = lam_scale / math.sqrt(max(T, F_dim))

        layers = []
        for k in range(K):
            c_mu_k = mu0 * (rho ** k)
            d_L_k = 1.0 / c_mu_k
            d_S_k = lam / c_mu_k

            layers.append(URCRPCALayer(c_mu_init=c_mu_k, d_L_init=d_L_k, d_S_init=d_S_k, rank=rank))
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _spectral_norm_batched(X: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.matrix_norm(X, ord=2)
        return norms.view(-1, 1, 1)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_at_end = False
        if X.ndim == 2:
            X = X.unsqueeze(0)
            squeeze_at_end = True
        if X.ndim != 3:
            raise ValueError(f"X must be (B, T, F) or (T, F), got shape {tuple(X.shape)}")

        X_norm = self._spectral_norm_batched(X)
        X_norm = torch.clamp(X_norm, min=1e-12)

        L = torch.zeros_like(X)
        S = torch.zeros_like(X)
        Y = torch.zeros_like(X)

        # Unroll K layers
        for layer in self.layers:
            L, S, Y = layer(X, L, S, Y, X_norm)

        if squeeze_at_end:
            L = L.squeeze(0)
            S = S.squeeze(0)
        return L, S

    def get_layer_params(self) -> list[dict]:
        return [
            {
                "layer": k,
                "c_mu": layer.c_mu.item(),
                "d_L": layer.d_L.item(),
                "d_S": layer.d_S.item(),
            }
            for k, layer in enumerate(self.layers)
        ]

    def __repr__(self) -> str:
        return (f"URCRPCA(K={self.K}, rank={self.rank}, T={self.T}, F={self.F_dim})")

