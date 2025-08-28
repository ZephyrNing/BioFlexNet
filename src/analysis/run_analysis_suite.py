import sys
sys.path.append("/rds/general/user/zn324/home/Flexible-Neurons-main")

import argparse
from pathlib import Path
from src.analysis.run_loader import RunLoader
from src.analysis.run_loader_image.gradient_flow import plot_grad_flow
from src.analysis.run_loader_image.intermediate_plot import IntermediateProcessViz
from src.analysis.run_loader_image.hessian import HessianAnalysis
from src.analysis.run_loader_image.loss_surface import PlotLossSurface
from src.training.dataset_select import get_dataset_obj
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter


# =========================
# Visualization helpers
# =========================
def visualize_loss_surface_2d(npy_path: Path, save_path: Path, smoothing_sigma=1):
    Z = np.load(npy_path)
    Z = gaussian_filter(Z, sigma=smoothing_sigma)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    c = ax.contourf(Z, levels=50, cmap="viridis")
    plt.colorbar(c)
    plt.title("Loss Surface (2D slice)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_loss_surface_3d(npy_path: Path, save_path: Path, smoothing_sigma=1, vmin=2, vmax=5):
    Z = np.load(npy_path)
    Z = gaussian_filter(Z, sigma=smoothing_sigma)
    steps = Z.shape[0]

    X = np.linspace(-1, 1, steps)
    Y = np.linspace(-1, 1, steps)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none',
                           antialiased=True)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Surface 3D')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_loss_surface_gif(npy_path: Path, gif_path: Path, frames=120, dpi=150, smoothing_sigma=1):
    Z = np.load(npy_path)
    Z = gaussian_filter(Z, sigma=smoothing_sigma)
    steps = Z.shape[0]

    X = np.linspace(-1, 1, steps)
    Y = np.linspace(-1, 1, steps)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.zaxis.label.set_size(12)
    ax.title.set_size(14)

    def make_frame(angle):
        ax.clear()
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
        ax.set_xlabel('Direction 1', fontsize=12)
        ax.set_ylabel('Direction 2', fontsize=12)
        ax.set_zlabel('Loss', fontsize=12)
        ax.set_title('Loss Surface 3D Rotation', fontsize=14)
        ax.view_init(elev=30, azim=angle)
        return fig

    gif_frames = []
    angles = np.linspace(0, 360, frames)

    for angle in angles:
        make_frame(angle)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        gif_frames.append(image)

    imageio.mimsave(gif_path, gif_frames, fps=30)
    plt.close(fig)


def visualize_hessian_density(eigen_path, weight_path, save_path):
    eigen = np.load(eigen_path)
    weight = np.load(weight_path)

    plt.figure(figsize=(6, 4))
    plt.plot(eigen, weight.real, marker='o', linestyle='-', linewidth=1)
    plt.yscale('log')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density (log scale)")
    plt.title("Hessian Spectrum Density")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# Core pipeline
# =========================
def run_analysis(checkpoint_dir: str):
    run_path = Path(checkpoint_dir)
    run_loader = RunLoader(run_path, whether_load_checkpoint=False)
    results_dir = Path("results") / run_path.name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[✓] Loaded model from: {checkpoint_dir}")

    # ---------- Load checkpoint (safe) ----------
    ckpt_dir = run_path / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("*.pth"))
    assert len(ckpt_files) > 0, f"No checkpoint found in {ckpt_dir}"

    # pick the latest checkpoint by mtime
    ckpt_file = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    save_content = torch.load(ckpt_file, map_location=run_loader.device)

    if "optimizer_state_dict" in save_content:
        print("[i] Skip loading optimizer_state_dict for analysis.")

    ckpt_sd = save_content["model_state_dict"]
    model_sd = run_loader.model.state_dict()

    # filter: only same-key & same-shape tensors
    loadable_sd = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable_sd[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape) if k in model_sd else "N/A"))

    missing, unexpected = run_loader.model.load_state_dict(loadable_sd, strict=False)
    print(f"[✓] Loaded {len(loadable_sd)}/{len(ckpt_sd)} tensors from {ckpt_file.name}")
    if skipped:
        print("[i] Skipped (shape mismatch):")
        for k, s_old, s_new in skipped[:12]:
            print(f"    - {k}: ckpt{s_old} -> model{s_new}")
        if len(skipped) > 12:
            print(f"    ... and {len(skipped)-12} more")

    run_loader.current_epoch = int(save_content.get("epoch", -1))
    run_loader.current_loss = float(save_content.get("loss", float('nan')))
    run_loader.model.eval()

    # ---------- 1) Gradient Flow ----------
    print("[...] Running gradient flow analysis")
    plot_grad_flow(run_loader, save_path=results_dir / "gradient_flow.png")

    # ---------- 2) Intermediate Representations ----------
    print("[...] Running intermediate layer visualization")
    dataset = get_dataset_obj(run_loader.config.dataset, "TRAIN")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    image_batch, _ = next(iter(dataloader))
    viz = IntermediateProcessViz(run_loader)
    viz.visualise(
        single_image_batch=image_batch,
        chunks=4,
        save_dir=results_dir / "intermediate_representations.png"
    )

    # ---------- 3) (Optional) Hessian ----------
    print("[...] Running Hessian analysis")
    hessian_analyser = HessianAnalysis(run_loader)
    hessian_out_dir = results_dir / "hessian"
    hessian_analyser.get_top_n_eigenvalues(save_dir=hessian_out_dir, top_n=5)
    hessian_analyser.get_trace_of_hessian(save_dir=hessian_out_dir, num_iter=1)
    print("[...] Visualizing Hessian spectrum density")
    iter_0 = hessian_out_dir / "hessian_iter_0"
    visualize_hessian_density(
        eigen_path=iter_0 / "density_eigen.npy",
        weight_path=iter_0 / "density_weight.npy",
        save_path=hessian_out_dir / "hessian_spectrum_density.png"
    )

    # ---------- 4) Loss Surface ----------
    print("[...] Running loss surface computation")
    loss_surface = PlotLossSurface(run_path)
    surface_path = results_dir / "loss_surface.npy"
    loss_surface.prepare_random_plane(save_as=surface_path)

    print("[...] Visualizing loss surface 2D")
    visualize_loss_surface_2d(
        npy_path=surface_path,
        save_path=results_dir / "loss_surface_2d.png",
        smoothing_sigma=1
    )

    print("[...] Visualizing loss surface 3D")
    visualize_loss_surface_3d(
        npy_path=surface_path,
        save_path=results_dir / "loss_surface_3d.png",
        smoothing_sigma=1 
    )

    print("[...] Visualizing loss surface rotating GIF (HD)")
    visualize_loss_surface_gif(
        npy_path=surface_path,
        gif_path=results_dir / "loss_surface.gif",
        frames=120,
        dpi=150,
        smoothing_sigma=1
    )

    print(f"[✓] All analysis complete. Saved to: {results_dir}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full analysis suite on a trained checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    run_analysis(args.checkpoint_dir)
