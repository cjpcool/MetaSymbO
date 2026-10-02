
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Saving utilities
# -------------------------
def save_predictions_and_targets(
    target_young: np.ndarray,
    target_shear: np.ndarray,
    target_poisson: np.ndarray,
    pred_young: np.ndarray,
    pred_shear: np.ndarray,
    pred_poisson: np.ndarray,
    out_dir: str | Path = "predictions_export",
    prefix: str = "metasymbo"
) -> Tuple[Path, Path]:
    """
    Save ground-truth and predicted arrays to both NPZ and CSV.

    Returns
    -------
    (npz_path, csv_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{prefix}_predictions.npz"
    np.savez_compressed(
        npz_path,
        target_young=target_young,
        target_shear=target_shear,
        target_poisson=target_poisson,
        pred_young=pred_young,
        pred_shear=pred_shear,
        pred_poisson=pred_poisson,
    )

    # CSV with columns: property, y_true, y_pred
    csv_path = out_dir / f"{prefix}_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("property,y_true,y_pred\n")
        for y_t, y_p in zip(target_young, pred_young):
            f.write(f"young,{y_t},{y_p}\n")
        for y_t, y_p in zip(target_shear, pred_shear):
            f.write(f"shear,{y_t},{y_p}\n")
        for y_t, y_p in zip(target_poisson, pred_poisson):
            f.write(f"poisson,{y_t},{y_p}\n")

    return npz_path, csv_path


# -------------------------
# Metrics
# -------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


# -------------------------
# Parity plotting
# -------------------------
def _nature_like_axes(ax: plt.Axes) -> None:
    """A minimal, 'Nature'-style aesthetic without external dependencies."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=10)
    # ax.grid(True, which='major', linewidth=0.6, alpha=0.2)
    # ax.grid(True, which='minor', linewidth=0.4, alpha=0.1)
    ax.minorticks_on()


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    xlabel: str = "Ground truth",
    ylabel: str = "Prediction",
    ax: Optional[plt.Axes] = None,
    point_size: float = 18.0,
    alpha: float = 0.7,
    identity_line: bool = True,
    annotate_stats: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Make a parity plot (each point is one sample; diagonal is y=x). 
    Computes and displays R^2, MAE, and RMSE.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    # Aesthetic tuning
    _nature_like_axes(ax)

    # Compute bounds with a small margin
    data_min = float(np.min([y_true.min(), y_pred.min()]))
    data_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 1e-6 * (data_max - data_min if data_max > data_min else 1.0)
    lo, hi = data_min - pad, data_max + pad

    # Scatter
    sc = ax.scatter(y_true, y_pred, s=point_size, alpha=alpha, linewidths=0.4, edgecolors='white')

    # Identity line
    if identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, pad=8)

    # Stats box
    if annotate_stats:
        r2 = r2_score(y_true, y_pred)
        _mae = mae(y_true, y_pred)
        _rmse = rmse(y_true, y_pred)
        text = rf"$R^2={r2:.3f}$" + "\n" + rf"$\mathrm{{MAE}}={_mae:.3g}$" + "\n" + rf"$\mathrm{{RMSE}}={_rmse:.3g}$"
        ax.text(
            0.04, 0.96, text, transform=ax.transAxes,
            va='top', ha='left', fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.08, linewidth=0.5)
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=512, bbox_inches="tight")

    return ax


def plot_all_parity(
    target_young: np.ndarray, pred_young: np.ndarray,
    target_shear: np.ndarray, pred_shear: np.ndarray,
    target_poisson: np.ndarray, pred_poisson: np.ndarray,
    titles: Tuple[str, str, str] = ("Young's Modulus", "Shear Modulus", "Poisson's Ratio"),
    save_path: Optional[str | Path] = None,
) -> None:
    """Create a 1x3 panel of parity plots for the three properties."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    plot_parity(target_young, pred_young, title=titles[0], ax=axes[0])
    plot_parity(target_shear, pred_shear, title=titles[1], ax=axes[1])
    plot_parity(target_poisson, pred_poisson, title=titles[2], ax=axes[2])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the saved .npz file (example path, adapt as needed)
    npz_path = "predictions_export/vae_cond_128_beta001_dis_same_100_frac_young_predictions.npz"
    data = np.load(npz_path)

    target_young = data["target_young"]
    target_shear = data["target_shear"]
    target_poisson = data["target_poisson"]
    pred_young = data["pred_young"]
    pred_shear = data["pred_shear"]
    pred_poisson = data["pred_poisson"]

    plot_parity(target_young, pred_young, title="Young's Modulus", save_path="figs/parity_young.png")
    plot_parity(target_shear, pred_shear, title="Shear Modulus", save_path="figs/parity_shear.png")
    plot_parity(target_poisson, pred_poisson, title="Poisson's Ratio", save_path="figs/parity_poisson.png")

    # 2b) Or one 1x3 multi-panel figure
    plot_all_parity(
        target_young, pred_young,
        target_shear, pred_shear,
        target_poisson, pred_poisson,
        save_path="figs/parity_all.png"
)