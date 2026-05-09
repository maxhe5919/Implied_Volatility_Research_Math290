import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
from config import logger, MONEYNESS_NODES


# --- MODULE 5: 3D VISUALIZATION (GIF GENERATOR) ---
def animate_volatility_surface(M, L, S, timestamps, dynamic_dte_nodes, filename="vol_surface_rpca.gif"):
    logger.info("MODULE 5: Generating 3D Volatility Surface Animation.")

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Use dynamic DTE nodes for the Y axis
    grid_x, grid_y = np.meshgrid(MONEYNESS_NODES, dynamic_dte_nodes)

    def update_graph(num):
        ax1.clear();
        ax2.clear();
        ax3.clear()

        Z_M = M[num].reshape(len(dynamic_dte_nodes), len(MONEYNESS_NODES))
        Z_L = L[num].reshape(len(dynamic_dte_nodes), len(MONEYNESS_NODES))
        Z_S = S[num].reshape(len(dynamic_dte_nodes), len(MONEYNESS_NODES))

        ax1.plot_surface(grid_x, grid_y, Z_M, cmap='viridis', edgecolor='none')
        ax1.set_title(f"Original Market (M)\nTime: {timestamps[num].time()}")
        ax1.set_zlim(np.nanmin(M), np.nanmax(M))

        ax2.plot_surface(grid_x, grid_y, Z_L, cmap='plasma', edgecolor='none')
        ax2.set_title("Background Structure (L)\nSmooth Level & Skew")
        ax2.set_zlim(np.nanmin(L), np.nanmax(L))

        ax3.plot_surface(grid_x, grid_y, Z_S, cmap='coolwarm', edgecolor='none')
        ax3.set_title("Market Anomalies (S)\n(Noise & Spikes)")
        ax3.set_zlim(np.nanmin(S), np.nanmax(S))

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('DTE (Days)')
            ax.set_zlabel('Custom IV')

        return fig,

    ani = animation.FuncAnimation(fig, update_graph, frames=len(timestamps), interval=100, blit=False)
    ani.save(filename, writer='pillow', fps=10)
    logger.info("Animation saved successfully!")


def plot_tensor_factors(factors, moneyness_nodes, dte_nodes, timestamps, anomaly_energy, filename="tensor_decomposition_results.png"):
    """
    Plots the 3 underlying modes of the Tensor Decomposition.
    This is the core proof of unsupervised latent factor discovery.
    """
    time_factors = factors[0]
    money_factors = factors[1]
    dte_factors = factors[2]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("PARAFAC Tensor Decomposition of Intraday Implied Volatility", fontsize=18, fontweight='bold')

    # --- PLOT 1: The Moneyness Mode (Strike Dimension) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(moneyness_nodes, money_factors[:, 0], label='Comp 1 (Level)', linewidth=2.5)
    ax1.plot(moneyness_nodes, money_factors[:, 1], label='Comp 2 (Skew)', linewidth=2.5, linestyle='--')
    ax1.plot(moneyness_nodes, money_factors[:, 2], label='Comp 3 (Curve)', linewidth=2.5, linestyle=':')
    ax1.set_title("Mode 2: Moneyness Loadings (The Smile/Skew)")
    ax1.set_xlabel("Moneyness (K/S)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: The DTE Mode (Term Structure Dimension) ---
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(dte_nodes, dte_factors[:, 0], label='Comp 1', marker='o')
    ax2.plot(dte_nodes, dte_factors[:, 1], label='Comp 2', marker='s')
    ax2.plot(dte_nodes, dte_factors[:, 2], label='Comp 3', marker='^')
    ax2.set_title("Mode 3: DTE Loadings (Term Structure)")
    ax2.set_xlabel("Days to Expiration (DTE)")
    ax2.set_xticks(dte_nodes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- PLOT 3: The Time Mode & Anomaly Energy ---
    ax3 = plt.subplot(2, 1, 2)

    # Normalize time factors for visual comparison
    df_time = pd.DataFrame(time_factors, columns=['Level', 'Skew', 'TermStruct'])
    df_norm = (df_time - df_time.mean()) / df_time.std()

    ax3.plot(timestamps, df_norm['Level'], color='blue', label='Factor 1 (Level Shift)')
    ax3.plot(timestamps, df_norm['Skew'], color='orange', linestyle='--', label='Factor 2 (Skew Tilt)')
    ax3.plot(timestamps, df_norm['TermStruct'], color='green', linestyle=':', label='Factor 3 (Term Struct Shift)')

    ax3.set_ylabel("Normalized Component Variation")
    ax3.set_xlabel("Local Time")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Twin axis for Anomaly Energy
    ax4 = ax3.twinx()
    ax4.plot(timestamps, anomaly_energy, color='red', alpha=0.6, linewidth=2, label='Anomaly Energy (Residuals²)')
    ax4.set_ylabel("Absolute Anomaly Energy", color='red')
    ax4.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_3d_surface_snapshot(tensor, background, residuals, moneyness_nodes, dte_nodes, timestamp_idx, timestamps, filename=None):
    """
    Takes a single slice of time (t) and plots the 3D surface separation.
    Proves: Raw Surface (X) = Background (L) + Sparse Anomalies (S)
    """
    X, Y = np.meshgrid(moneyness_nodes, dte_nodes)

    raw_slice = tensor[timestamp_idx, :, :].T
    bg_slice = background[timestamp_idx, :, :].T
    res_slice = residuals[timestamp_idx, :, :].T

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Surface Decomposition Snapshot at {timestamps[timestamp_idx]}", fontsize=16)

    # 1. Raw Tensor Slice
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, raw_slice, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title("Original IV Surface (X)")
    ax1.set_xlabel("Moneyness")
    ax1.set_ylabel("DTE")

    # 2. Background (Low Rank)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, bg_slice, cmap='plasma', edgecolor='none', alpha=0.8)
    ax2.set_title("Structural Consensus (Rank-3 Background)")
    ax2.set_xlabel("Moneyness")
    ax2.set_ylabel("DTE")

    # 3. Sparse Anomalies
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X, Y, res_slice, cmap='coolwarm', edgecolor='none', alpha=0.8)
    ax3.set_title("Localized Mispricings (Residuals)")
    ax3.set_xlabel("Moneyness")
    ax3.set_ylabel("DTE")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def animate_tensor_surface(tensor, background, residuals, moneyness_nodes, dte_nodes, timestamps, filename='tensor_decomposition_surface.gif'):
    """
    Generates a GIF of the 3D Tensor Decomposition over time.
    """
    print("Generating 3D Tensor Animation (this may take a minute)...")

    X, Y = np.meshgrid(moneyness_nodes, dte_nodes)

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        raw_slice = tensor[frame, :, :].T
        bg_slice = background[frame, :, :].T
        res_slice = residuals[frame, :, :].T

        # Fixed Z-limits so the surface doesn't wildly jump around
        z_min, z_max = np.nanmin(tensor), np.nanmax(tensor)

        ax1.plot_surface(X, Y, raw_slice, cmap='viridis', edgecolor='none', alpha=0.8)
        ax1.set_title(f"Raw IV Surface | Time: {timestamps[frame]}")
        ax1.set_zlim(z_min, z_max)

        ax2.plot_surface(X, Y, bg_slice, cmap='plasma', edgecolor='none', alpha=0.8)
        ax2.set_title("Background (Low Rank Consensus)")
        ax2.set_zlim(z_min, z_max)

        ax3.plot_surface(X, Y, res_slice, cmap='coolwarm', edgecolor='none', alpha=0.8)
        ax3.set_title("Anomalies (Sparse Residuals)")
        ax3.set_zlim(-np.nanmax(np.abs(residuals)), np.nanmax(np.abs(residuals)))

    # Animate every Nth frame to save rendering time if the dataset is huge
    step = max(1, len(timestamps) // 100)
    frames = np.arange(0, len(timestamps), step)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Animation saved to {filename}")