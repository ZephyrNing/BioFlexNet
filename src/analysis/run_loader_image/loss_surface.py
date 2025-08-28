# src/analysis/run_loader_image/loss_surface.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.analysis.run_loader import RunLoader
from src.training.dataset_select import get_dataset_obj
from src._loss_surface.loss_landscapes import metrics
from src._loss_surface.loss_landscapes import random_plane, linear_interpolation
from src.utils.device import select_device


def safe_load_state_dict(model: torch.nn.Module, state_dict: dict, *, verbose: bool = True) -> None:
    """
    只加载形状完全匹配的权重；不匹配的一律跳过并打印。
    不会触发 size mismatch 的异常。
    """
    model_sd = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            # 记录一下，方便你核对
            model_shape = model_sd[k].shape if k in model_sd else "N/A"
            skipped.append((k, tuple(v.shape), model_shape))

    # 只用可加载子集更新
    missing_keys = [k for k in model_sd.keys() if k not in state_dict]
    model.load_state_dict(loadable, strict=False)

    if verbose:
        print(f"[✓] Loaded {len(loadable)}/{len(state_dict)} tensors into model")
        if skipped:
            print("[i] Skipped (shape mismatch or missing in model):")
            for name, ckpt_shape, model_shape in skipped:
                print(f"    - {name}: ckpt{ckpt_shape} -> model{model_shape}")
        if missing_keys:
            print(f"[i] Model params missing from checkpoint: {len(missing_keys)} keys (left at init values)")


class PlotLossSurface:
    def __init__(
        self,
        run_folder,
        steps=100,
        distance=1,
        normalisation: str = "filter",
        criterion=torch.nn.CrossEntropyLoss(),
    ):
        # fmt: off
        torch.manual_seed(42)
        np.random.seed(42)
        self.steps = steps
        self.distance = distance
        self.normalisation = normalisation

        self.device = select_device()

        # 不自动加载 checkpoint，避免触发严格匹配
        self.run_loader_initial = RunLoader(run_folder, whether_load_checkpoint=False)
        self.run_loader_terminal = RunLoader(run_folder, whether_load_checkpoint=False)

        # 手动、**安全**地加载 checkpoint（仅模型；不加载 optimizer）
        ckpt_dir = Path(run_folder) / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("*.pth"))
        if len(ckpt_files) == 0:
            raise FileNotFoundError(f"No .pth found under: {ckpt_dir}")
        ckpt_path = ckpt_files[0]
        save_content = torch.load(ckpt_path, map_location=self.device)

        model_state = save_content.get("model_state_dict", {})
        if not model_state:
            raise KeyError(f"'model_state_dict' not found in checkpoint: {ckpt_path}")

        # ⭐ 关键：只加载形状匹配的权重
        safe_load_state_dict(self.run_loader_terminal.model, model_state, verbose=True)

        # 这些状态对画 loss surface 不是必须，但保留打印更直观
        self.run_loader_terminal.current_epoch = save_content.get("epoch", 0)
        self.run_loader_terminal.current_loss = save_content.get("loss", 0.0)

        batch_size = self.run_loader_initial.config.batch_size
        # fmt: on

        # -------- 构造一个小批数据用于评估 loss --------
        dataset = get_dataset_obj(self.run_loader_initial.config.dataset, "TRAIN")
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.x, self.y = next(iter(self.dataloader))
        self.x, self.y = self.x.to(self.device), self.y.to(self.device)
        self.metric = metrics.Loss(criterion, self.x, self.y)

    def prepare_random_plane(self, save_as=None):
        """Prepare the random plane data for both 2D/3D surface plotting."""
        self.loss_data_plane = random_plane(
            self.run_loader_terminal.model,
            self.metric,
            distance=self.distance,
            steps=self.steps,
            normalization=self.normalisation,
            deepcopy_model=False,
        )
        if save_as:
            np.save(save_as, self.loss_data_plane)


if __name__ == "__main__":
    pass
