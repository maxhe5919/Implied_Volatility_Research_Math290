from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unrolled_rc_rpca import URCRPCA
from models.rc_rpca import RCRPCA, RCRPCAConfig
from training.dataset import SyntheticIVDataset, PERTURBATION_CLASSES


@torch.no_grad()
def score_urcrpca(model: URCRPCA, X_batch: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    X_batch = X_batch.to(device)
    _, S_pred = model(X_batch)
    return S_pred.abs().cpu().numpy()


def score_rcrpca(model: RCRPCA, X_batch_np: np.ndarray) -> np.ndarray:
    out = np.empty_like(X_batch_np)
    for i in range(X_batch_np.shape[0]):
        _, S = model.fit_window(X_batch_np[i])
        out[i] = np.abs(S)
    return out

def compute_auroc_per_window(scores: np.ndarray, mask: np.ndarray) -> float | None:
    y_true = mask.flatten().astype(np.int8)
    y_score = scores.flatten().astype(np.float64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    return roc_auc_score(y_true, y_score)


def compute_anomaly_energy(scores: np.ndarray) -> float:
    return float(np.sqrt(np.mean(scores ** 2)))

@dataclass
class EvalResult:
    model_name: str
    ptype: str
    magnitude: float
    n_windows: int
    auroc_mean: float | None
    auroc_std: float | None
    anomaly_energy_mean: float
    anomaly_energy_std: float


def evaluate_model(name: str,
                   score_fn,
                   ds: SyntheticIVDataset,
                   batch_size: int = 64,
                   device: torch.device | None = None) -> list[EvalResult]:
    groups = defaultdict(list)
    for idx in range(len(ds)):
        meta = ds.get_metadata(idx)
        groups[(meta["ptype"], meta["magnitude"])].append(idx)

    results = []
    n_groups = len(groups)
    for gi, ((ptype, mag), indices) in enumerate(sorted(groups.items())):
        print(f"  starting group {gi + 1}/{n_groups}: ptype={ptype} mag={mag}", flush=True)
        per_window_aurocs = []
        per_window_energies = []

        n_batches = (len(indices) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            chunk = indices[bi * batch_size:(bi + 1) * batch_size]
            X_chunk = np.stack([ds.X_dirty[i] for i in chunk])
            mask_chunk = np.stack([ds.mask[i] for i in chunk])
            scores = score_fn(X_chunk)

            for s, m in zip(scores, mask_chunk):
                auroc = compute_auroc_per_window(s, m)
                if auroc is not None:
                    per_window_aurocs.append(auroc)
                per_window_energies.append(compute_anomaly_energy(s))

        auroc_mean = float(np.mean(per_window_aurocs)) if per_window_aurocs else None
        auroc_std = float(np.std(per_window_aurocs)) if per_window_aurocs else None

        results.append(EvalResult(
            model_name=name,
            ptype=ptype,
            magnitude=mag,
            n_windows=len(indices),
            auroc_mean=auroc_mean,
            auroc_std=auroc_std,
            anomaly_energy_mean=float(np.mean(per_window_energies)),
            anomaly_energy_std=float(np.std(per_window_energies)),
        ))
        print(f"  [{gi+1}/{n_groups}] {name:12s} {ptype} mag={mag:.1f}  "
              f"N={len(indices):4d}  "
              f"AUROC={'--' if auroc_mean is None else f'{auroc_mean:.3f}'}  "
              f"energy={results[-1].anomaly_energy_mean:.4f}")

    return results

def plot_auroc_vs_magnitude(results: list[EvalResult], out_path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    by_ptype = defaultdict(list)
    for r in results:
        by_ptype[r.ptype].append(r)

    for ax, ptype in zip(axes, PERTURBATION_CLASSES):
        ax.set_title(f"{ptype}")
        ax.set_xlabel("Magnitude (sigma_noise units)")
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5,
                   label='_random' if ax is not axes[0] else 'Random')
        ax.grid(True, alpha=0.3)
        by_model = defaultdict(list)
        for r in by_ptype.get(ptype, []):
            by_model[r.model_name].append(r)

        for model_name, rs in by_model.items():
            rs = sorted(rs, key=lambda r: r.magnitude)
            mags = [r.magnitude for r in rs]
            aurocs = [r.auroc_mean for r in rs]
            stds = [r.auroc_std for r in rs]
            if all(a is None for a in aurocs):
                ax.text(0.5, 0.7, "AUROC undefined\n(see null-case plot)",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='gray')
                continue
            valid = [(m, a, s) for m, a, s in zip(mags, aurocs, stds) if a is not None]
            if valid:
                m_v, a_v, s_v = zip(*valid)
                ax.errorbar(m_v, a_v, yerr=s_v, marker='o', label=model_name,
                            capsize=3, linewidth=2)
        if ax is axes[0]:
            ax.set_ylabel("Pixel-level AUROC")
            ax.legend(loc='lower right', fontsize=9)

    fig.suptitle("Detection characteristic: AUROC vs perturbation magnitude", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_null_case_energy(results: list[EvalResult], out_path: Path):
    import matplotlib.pyplot as plt

    by_model_class = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_model_class[r.model_name][r.ptype].append(r.anomaly_energy_mean)

    models = list(by_model_class.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(PERTURBATION_CLASSES))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        means = [
            float(np.mean(by_model_class[model].get(p, [np.nan])))
            for p in PERTURBATION_CLASSES
        ]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               label=model, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(PERTURBATION_CLASSES)
    ax.set_ylabel("Mean ||S_pred|| (RMS)")
    ax.set_title("Anomaly energy by perturbation class\n"
                 "(P2/P4: smaller = better; P1/P3: larger = stronger detection)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def write_csv(results: list[EvalResult], out_path: Path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ptype", "magnitude", "n_windows",
                    "auroc_mean", "auroc_std",
                    "anomaly_energy_mean", "anomaly_energy_std"])
        for r in results:
            w.writerow([
                r.model_name, r.ptype, r.magnitude, r.n_windows,
                "" if r.auroc_mean is None else f"{r.auroc_mean:.6f}",
                "" if r.auroc_std is None else f"{r.auroc_std:.6f}",
                f"{r.anomaly_energy_mean:.6f}",
                f"{r.anomaly_energy_std:.6f}",
            ])
    print(f"Saved {out_path}")


def main():
    print("MAIN START", flush=True)
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directory with best.pt and config.json")
    p.add_argument("--npz", required=True, help="Path to synthetic dataset .npz")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--include-converged-baseline", action="store_true", default=True, help="Also evaluate classical RC-RPCA run-to-convergence")
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
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
        print(f"Loaded best.pt (val_loss={ckpt.get('val_loss')})")
    elif final_path.exists():
        ckpt = torch.load(final_path, map_location=device)
        print(f"WARN: best.pt missing, falling back to final.pt")
    else:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded U-RC-RPCA from {run_dir}/best.pt "
          f"(K={K}, rank={rank}, val_loss={ckpt.get('val_loss')})")

    ds = SyntheticIVDataset(args.npz, split=args.split)
    print(f"Eval split: {args.split}, N={len(ds)}")

    def score_urcrpca_inner(X_np):
        X_torch = torch.from_numpy(X_np).to(torch.float32)
        return score_urcrpca(model, X_torch, device)

    cls_K_model = RCRPCA(RCRPCAConfig(rank=rank, K=K))
    def score_classical_K(X_np):
        return score_rcrpca(cls_K_model, X_np)

    cls_conv_model = RCRPCA(RCRPCAConfig(rank=rank, K=None,
                                          tol=1e-7, max_iter=500))
    def score_classical_conv(X_np):
        return score_rcrpca(cls_conv_model, X_np)

    print("\n Evaluating U-RC-RPCA ")
    res_urc = evaluate_model("U-RC-RPCA", score_urcrpca_inner, ds,
                             batch_size=args.batch_size, device=device)

    print(f"\n Evaluating classical RC-RPCA at K={K}  ")
    res_K = evaluate_model(f"RC-RPCA(K={K})", score_classical_K, ds,
                           batch_size=args.batch_size, device=device)

    all_results = res_urc + res_K
    if args.include_converged_baseline:
        print("\n Evaluating classical RC-RPCA run-to-convergence ")
        res_conv = evaluate_model("RC-RPCA(conv)", score_classical_conv, ds,
                                  batch_size=args.batch_size, device=device)
        all_results += res_conv

    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    write_csv(all_results, out_dir / "auroc_table.csv")
    plot_auroc_vs_magnitude(all_results, out_dir / "auroc_vs_magnitude.png")
    plot_null_case_energy(all_results, out_dir / "null_case_energy.png")
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
