from __future__ import annotations

from pathlib import Path
import json
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .metrics import probability_map, sigmoid_map


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dump_json(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_prediction_triptych(
    image_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    logits: torch.Tensor,
    use_probability: bool,
    baseline_logits: torch.Tensor | None,
    path: str | Path,
) -> None:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)

    image = (image_tensor * std + mean).clamp(0.0, 1.0).detach().cpu()
    image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    image_pil = Image.fromarray(image)

    target = target_tensor.squeeze(0).detach().cpu().numpy()
    target = target - target.min()
    target = (target / (target.max() + 1e-8) * 255.0).astype(np.uint8)
    target_pil = Image.fromarray(target, mode="L").convert("RGB")

    pred_tensor = probability_map(logits.unsqueeze(0)) if use_probability else sigmoid_map(logits.unsqueeze(0))
    pred = pred_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred = pred - pred.min()
    pred = (pred / (pred.max() + 1e-8) * 255.0).astype(np.uint8)
    pred_pil = Image.fromarray(pred, mode="L").convert("RGB")

    panels = [image_pil, target_pil]

    if baseline_logits is not None:
        baseline_tensor = probability_map(baseline_logits.unsqueeze(0)) if use_probability else sigmoid_map(baseline_logits.unsqueeze(0))
        baseline = baseline_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
        baseline = baseline - baseline.min()
        baseline = (baseline / (baseline.max() + 1e-8) * 255.0).astype(np.uint8)
        baseline_pil = Image.fromarray(baseline, mode="L").convert("RGB")
        panels.append(baseline_pil)

    panels.append(pred_pil)

    canvas = Image.new("RGB", (image_pil.width * len(panels), image_pil.height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (image_pil.width * index, 0))
    canvas.save(path)


def save_loss_curve(path: str | Path, records: list[dict]) -> None:
    matplotlib.use("Agg")
    if not records:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
        ax.set_title("Training Curves")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    train_records = [item for item in records if item["split"] == "train"]
    val_records = [item for item in records if item["split"] == "val"]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    if train_records:
        ax.plot(
            [item["step"] for item in train_records],
            [max(item["loss"], 1e-12) for item in train_records],
            color="#2266cc",
            label="train loss",
            linewidth=1.8,
        )
    if val_records:
        ax.plot(
            [item["step"] for item in val_records],
            [max(item["loss"], 1e-12) for item in val_records],
            color="#dc4646",
            label="val loss",
            linewidth=1.8,
        )
    ax.set_title("Training Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    if train_records:
        ax.plot(
            [item["step"] for item in train_records],
            [item["cc"] for item in train_records],
            color="#2266cc",
            label="train CC",
            linewidth=1.8,
        )
    if val_records:
        ax.plot(
            [item["step"] for item in val_records],
            [item["cc"] for item in val_records],
            color="#dc4646",
            label="val CC",
            linewidth=1.8,
        )
    ax.set_title("CC Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("CC")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    cc_path = Path(path).with_name("cc_curve.png")
    fig.savefig(cc_path)
    plt.close(fig)
