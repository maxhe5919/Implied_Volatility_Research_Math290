"""
Training loop for U-RC-RPCA.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unrolled_rc_rpca import URCRPCA
from training.dataset import SyntheticIVDataset

@dataclass
class TrainConfig:
    K: int = 8
    rank: int = 3
    T: int = 60
    # M * D = 21 * 5
    F_dim: int = 105
    npz_path: str = "data/synthetic.npz"
    batch_size: int = 32
    num_workers: int = 0

    lr: float = 1e-3
    epochs: int = 60
    alpha: float = 2.0
    grad_clip: float = 1.0
    weight_decay: float = 0.0

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    seed: int = 42
    device: str = "auto"
    run_name: str = "urc_K8"
    output_root: str = "runs"
    log_every: int = 10

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

def reconstruction_loss(L_pred: torch.Tensor, S_pred: torch.Tensor,
                        L_true: torch.Tensor, S_true: torch.Tensor,
                        alpha: float = 1.0) -> tuple[torch.Tensor, dict]:

    L_loss = (L_pred - L_true).pow(2).sum(dim=(-1, -2)).mean()
    S_loss = (S_pred - S_true).pow(2).sum(dim=(-1, -2)).mean()
    total = L_loss + alpha * S_loss
    return total, {"L_loss": L_loss.item(), "S_loss": S_loss.item(), "total": total.item()}


def cfg_initial(k: int, name: str, F_dim: int = 105, T: int = 60,
                rho: float = 1.5, mu0: float = 1.25) -> float:
    import math
    c_mu = mu0 * (rho ** k)
    if name == "c_mu":
        return c_mu
    if name == "d_L":
        return 1.0 / c_mu
    if name == "d_S":
        lam = 1.0 / math.sqrt(max(T, F_dim))
        return lam / c_mu
    raise ValueError(name)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    cfg: TrainConfig,
                    device: torch.device,
                    epoch: int) -> dict:
    model.train()
    n_batches = len(loader)
    running = {"total": 0.0, "L_loss": 0.0, "S_loss": 0.0}
    n_skipped = 0

    for i, (X, L_true, S_true) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        L_true = L_true.to(device, non_blocking=True)
        S_true = S_true.to(device, non_blocking=True)

        if not torch.isfinite(X).all():
            n_skipped += 1
            print(f"  WARN: epoch {epoch} batch {i} has non-finite input; skipping")
            continue

        try:
            L_pred, S_pred = model(X)
        except torch._C._LinAlgError as e:
            n_skipped += 1
            print(f"  WARN: epoch {epoch} batch {i} SVD failed: {e}")
            print(f"        X stats: min={X.min().item():.3e} "
                  f"max={X.max().item():.3e} "
                  f"||X||_F={X.norm().item():.3e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not (torch.isfinite(L_pred).all() and torch.isfinite(S_pred).all()):
            n_skipped += 1
            print(f"  WARN: epoch {epoch} batch {i} non-finite model output; skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss, components = reconstruction_loss(
            L_pred, S_pred, L_true, S_true, alpha=cfg.alpha
        )

        if not torch.isfinite(loss):
            n_skipped += 1
            print(f"  WARN: epoch {epoch} batch {i} non-finite loss; skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        for k in running:
            running[k] += components[k]

        if (i + 1) % cfg.log_every == 0:
            print(f"  epoch {epoch} batch {i+1}/{n_batches} "
                  f"loss={components['total']:.4e} "
                  f"(L={components['L_loss']:.4e}, S={components['S_loss']:.4e})")

    n_used = n_batches - n_skipped
    if n_used == 0:
        return {k: float("nan") for k in running}
    if n_skipped > 0:
        print(f"  epoch {epoch} skipped {n_skipped}/{n_batches} batches")
    return {k: v / n_used for k, v in running.items()}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             cfg: TrainConfig,
             device: torch.device) -> dict:
    model.eval()
    n_batches = len(loader)
    running = {"total": 0.0, "L_loss": 0.0, "S_loss": 0.0}
    for X, L_true, S_true in loader:
        X = X.to(device, non_blocking=True)
        L_true = L_true.to(device, non_blocking=True)
        S_true = S_true.to(device, non_blocking=True)
        L_pred, S_pred = model(X)
        _, components = reconstruction_loss(
            L_pred, S_pred, L_true, S_true, alpha=cfg.alpha
        )
        for k in running:
            running[k] += components[k]
    return {k: v / n_batches for k, v in running.items()}

def train(cfg: TrainConfig) -> Path:
    device = cfg.resolve_device()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    train_ds = SyntheticIVDataset(cfg.npz_path, split="train")
    val_ds = SyntheticIVDataset(cfg.npz_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=(device.type == "cuda"))
    print(f"Train N={len(train_ds)}, Val N={len(val_ds)}")
    print(f"T={train_ds.T}, F={train_ds.F}")

    if train_ds.T != cfg.T or train_ds.F != cfg.F_dim:
        raise ValueError(
            f"Config T={cfg.T}, F_dim={cfg.F_dim} does not match "
            f"data T={train_ds.T}, F={train_ds.F}. Update config."
        )

    model = URCRPCA(K=cfg.K, rank=cfg.rank, T=cfg.T, F_dim=cfg.F_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model}, {n_params} learnable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.output_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    init_val = evaluate(model, val_loader, cfg, device)
    print(f"\n[init] val total={init_val['total']:.4e} "
          f"(L={init_val['L_loss']:.4e}, S={init_val['S_loss']:.4e})")
    print("       (this is classical RC-RPCA at K={} -- the baseline to beat)\n"
          .format(cfg.K))

    history = {"train": [], "val": [], "init_val": init_val}
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch)
        val_metrics = evaluate(model, val_loader, cfg, device)
        epoch_time = time.time() - t0

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if best_val == float("inf"):
            improved = math.isfinite(val_metrics["total"])
        else:
            improvement = (best_val - val_metrics["total"]) / best_val
            improved = improvement > cfg.min_delta

        if improved:
            best_val = val_metrics["total"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
                "config": asdict(cfg),
            }, out_dir / "best.pt")
            star = " *"
        else:
            epochs_since_improve += 1
            star = ""

        print(f"epoch {epoch:3d} ({epoch_time:5.1f}s) "
              f"train={train_metrics['total']:.4e} "
              f"val={val_metrics['total']:.4e}{star}")

        if epochs_since_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {cfg.patience} epochs).")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete. Total time {total_time:.1f}s. "
          f"Best val={best_val:.4e} at epoch {best_epoch}.")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
    }, out_dir / "final.pt")

    layer_params = model.get_layer_params()
    with open(out_dir / "layer_params.json", "w") as f:
        json.dump({
            "init": [
                {"layer": k,
                 "c_mu": cfg_initial(k, "c_mu", F_dim=cfg.F_dim, T=cfg.T),
                 "d_L": cfg_initial(k, "d_L", F_dim=cfg.F_dim, T=cfg.T),
                 "d_S": cfg_initial(k, "d_S", F_dim=cfg.F_dim, T=cfg.T)}
                for k in range(cfg.K)
            ],
            "trained": layer_params,
        }, f, indent=2)

    return out_dir


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--F-dim", type=int, default=105)
    p.add_argument("--npz", type=str, default="data/synthetic.npz",
                   dest="npz_path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-root", type=str, default="runs")
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = f"urc_K{args.K}_alpha{args.alpha}"
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    out_dir = train(cfg)
    print(f"Outputs written to {out_dir}")
