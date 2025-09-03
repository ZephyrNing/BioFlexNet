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
import pickle



def visualize_loss_surface(npy_path: Path, save_path: Path):
    Z = np.load(npy_path)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    c = ax.contourf(Z, levels=50, cmap="viridis")
    plt.colorbar(c)
    plt.title("Loss Surface (2D slice)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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



def run_analysis(checkpoint_dir: str):
    run_path = Path(checkpoint_dir)
    run_loader = RunLoader(run_path, whether_load_checkpoint=False)
    results_dir = Path("results") / run_path.name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[✓] Loaded model from: {checkpoint_dir}")

    # ⭐ 手动加载 checkpoint，strict=False
    ckpt_dir = run_path / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("*.pth"))
    save_content = torch.load(ckpt_files[0], map_location=run_loader.device)
    run_loader.model.load_state_dict(save_content["model_state_dict"], strict=False)
    run_loader.optimizer.load_state_dict(save_content["optimizer_state_dict"])
    run_loader.current_epoch, run_loader.current_loss = save_content["epoch"], save_content["loss"]

    run_loader.model.eval()

    # 后续保持不变



    print(f"[\u2713] Loaded model from: {checkpoint_dir}")

    # 1. Gradient Flow
    print("[...] Running gradient flow analysis")
    plot_grad_flow(run_loader, save_path=results_dir / "gradient_flow.png")

    # 2. Intermediate Representations
    print("[...] Running intermediate layer visualization")
    dataset = get_dataset_obj(run_loader.config.dataset, "TRAIN")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    image_batch, _ = next(iter(dataloader))
    viz = IntermediateProcessViz(run_loader)
    viz.visualise(single_image_batch=image_batch, chunks=4,
                  save_dir=results_dir / "intermediate_representations.png")

    # 3. Hessian Analysis
    print("[...] Running Hessian analysis")
    hessian_analyser = HessianAnalysis(run_loader)
    hessian_out_dir = results_dir / "hessian"
    hessian_analyser.get_top_n_eigenvalues(save_dir=hessian_out_dir, top_n=5)
    hessian_analyser.get_trace_of_hessian(save_dir=hessian_out_dir, num_iter=1)

    # 3.1 Visualize Hessian density
    print("[...] Visualizing Hessian spectrum density")
    iter_0 = hessian_out_dir / "hessian_iter_0"
    visualize_hessian_density(
        eigen_path=iter_0 / "density_eigen.npy",
        weight_path=iter_0 / "density_weight.npy",
        save_path=hessian_out_dir / "hessian_spectrum_density.png"
    )

    # 4. Loss Surface
    print("[...] Running loss surface computation")
    loss_surface = PlotLossSurface(run_path)
    surface_path = results_dir / "loss_surface.npy"
    loss_surface.prepare_random_plane(save_as=surface_path)

    print("[...] Visualizing loss surface")
    visualize_loss_surface(npy_path=surface_path, save_path=results_dir / "loss_surface_plot.png")

    print(f"[\u2713] All analysis complete. Saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full analysis suite on a trained checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory")
    args = parser.parse_args()

    torch.manual_seed(42)
    run_analysis(args.checkpoint_dir)
