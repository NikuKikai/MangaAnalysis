from __future__ import annotations

from pathlib import Path
import shutil

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import Config
from .datasets import FiWIDataset, SaliconDataset, build_fiwi_splits
from .losses import CombinedSaliencyLoss
from .metrics import cc_metric, kld_metric, mse_metric
from .model import SaliencyUNet
from .utils import dump_json, ensure_dir, save_loss_curve, save_prediction_triptych, set_seed


def create_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if config.dataset.name == "salicon":
        train_ds = SaliconDataset(
            root=config.dataset.root,
            split=config.dataset.train_split or "train",
            input_size=config.dataset.input_size,
            limit=config.dataset.train_limit,
        )
        val_ds = SaliconDataset(
            root=config.dataset.root,
            split=config.dataset.val_split or "val",
            input_size=config.dataset.input_size,
            limit=config.dataset.val_limit,
        )
        test_ds = None
    elif config.dataset.name == "fiwi":
        splits = build_fiwi_splits(
            root=config.dataset.root,
            seed=config.experiment.seed,
            train_ratio=config.dataset.train_ratio or 0.7,
            val_ratio=config.dataset.val_ratio or 0.15,
            test_ratio=config.dataset.test_ratio or 0.15,
        )
        val_ds = FiWIDataset(
            splits["val"],
            config.dataset.input_size,
            config.dataset.val_limit,
            config.dataset.fixation_blur_sigma or 35.0,
        )
        test_ds = FiWIDataset(
            splits["test"],
            config.dataset.input_size,
            config.dataset.test_limit,
            config.dataset.fixation_blur_sigma or 35.0,
        )
        train_ds = FiWIDataset(
            splits["train"],
            config.dataset.input_size,
            config.dataset.train_limit,
            config.dataset.fixation_blur_sigma or 35.0,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.loader.batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.loader.batch_size,
            shuffle=False,
            num_workers=config.loader.num_workers,
        )
    return train_loader, val_loader, test_loader


def create_model(config: Config) -> SaliencyUNet:
    model = SaliencyUNet(
        encoder_pretrained=config.model.encoder_pretrained,
        base_channels=config.model.base_channels,
    )
    if config.training.init_checkpoint and Path(config.training.init_checkpoint).exists():
        checkpoint = torch.load(config.training.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.freeze_encoder_stages(config.model.freeze_stages)
    return model


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def use_probability_metrics(config: Config) -> bool:
    return config.loss.kld_weight > 0.0 and config.loss.bce_weight == 0.0 and config.loss.mse_weight == 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedSaliencyLoss,
    device: torch.device,
    optimizer: AdamW | None,
    use_probability: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    totals = {"loss": 0.0, "kld": 0.0, "bce": 0.0, "mse": 0.0, "cc": 0.0}
    batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        target_prob = batch["target_prob"].to(device)
        target_max = batch["target_max"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss, parts = criterion(logits, target_prob, target_max)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        totals["loss"] += float(loss.detach())
        totals["kld"] += parts["kld"]
        totals["bce"] += parts["bce"]
        totals["mse"] += mse_metric(logits, target_prob if use_probability else target_max, use_probability)
        totals["cc"] += cc_metric(logits, target_prob if use_probability else target_max, use_probability)
        batches += 1

    if batches == 0:
        return totals
    return {key: value / batches for key, value in totals.items()}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedSaliencyLoss,
    device: torch.device,
    use_probability: bool,
) -> dict[str, float]:
    return run_epoch(model, loader, criterion, device, None, use_probability)


def save_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    sample_count: int,
    use_probability: bool,
    baseline_model: nn.Module | None,
) -> None:
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = (batch["target_prob"] if use_probability else batch["target_max"]).to(device)
            logits = model(image)
            baseline_logits = baseline_model(image) if baseline_model is not None else None
            names = batch["name"]
            for index in range(image.shape[0]):
                save_prediction_triptych(
                    image[index],
                    target[index],
                    logits[index],
                    use_probability,
                    baseline_logits[index] if baseline_logits is not None else None,
                    output_dir / f"{names[index]}.png",
                )
                saved += 1
                if saved >= sample_count:
                    return


def train(config: Config, config_path: str) -> dict[str, dict[str, float]]:
    set_seed(config.experiment.seed)
    device = resolve_device(config.training.device)
    use_probability = use_probability_metrics(config)
    train_loader, val_loader, test_loader = create_loaders(config)

    model = create_model(config).to(device)
    baseline_model = None
    if config.training.compare_checkpoint and Path(config.training.compare_checkpoint).exists():
        baseline_model = SaliencyUNet(
            encoder_pretrained=False,
            base_channels=config.model.base_channels,
        ).to(device)
        checkpoint = torch.load(config.training.compare_checkpoint, map_location=device)
        baseline_model.load_state_dict(checkpoint["model"])
    criterion = CombinedSaliencyLoss(config.loss.kld_weight, config.loss.bce_weight, config.loss.mse_weight)
    optimizer = AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    output_dir = ensure_dir(config.experiment.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    samples_dir = ensure_dir(output_dir / "samples")
    shutil.copy2(config_path, output_dir / "config.toml")

    history: dict[str, object] = {"records": []}
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    global_step = 0
    running = {"loss": 0.0, "kld": 0.0, "bce": 0.0, "mse": 0.0, "cc": 0.0, "count": 0}

    for epoch in range(1, config.training.epochs + 1):
        model.train(True)
        for batch in train_loader:
            image = batch["image"].to(device)
            target_prob = batch["target_prob"].to(device)
            target_max = batch["target_max"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss, parts = criterion(logits, target_prob, target_max)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running["loss"] += float(loss.detach())
            running["kld"] += parts["kld"]
            running["bce"] += parts["bce"]
            running["mse"] += mse_metric(logits, target_prob if use_probability else target_max, use_probability)
            running["cc"] += cc_metric(logits, target_prob if use_probability else target_max, use_probability)
            running["count"] += 1

            if global_step % config.training.eval_interval_steps == 0:
                train_record = {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "train",
                "loss": running["loss"] / running["count"],
                "kld": running["kld"] / running["count"],
                "bce": running["bce"] / running["count"],
                "mse": running["mse"] / running["count"],
                "cc": running["cc"] / running["count"],
                }
                history["records"].append(train_record)
                val_metrics = evaluate(model, val_loader, criterion, device, use_probability)
                val_record = {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "val",
                    **val_metrics,
                }
                history["records"].append(val_record)
                dump_json(output_dir / "metrics.json", history)
                save_loss_curve(output_dir / "loss_curve.png", history["records"])
                running = {"loss": 0.0, "kld": 0.0, "bce": 0.0, "mse": 0.0, "cc": 0.0, "count": 0}

        if epoch % config.training.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "step": global_step}, checkpoints_dir / f"epoch_{epoch:03d}.pt")

    val_metrics = evaluate(model, val_loader, criterion, device, use_probability)
    history["final_val"] = val_metrics

    if test_loader is not None:
        history["test"] = evaluate(model, test_loader, criterion, device, use_probability)
        save_samples(model, test_loader, device, ensure_dir(samples_dir / "test"), config.training.sample_count, use_probability, baseline_model)
    else:
        save_samples(model, val_loader, device, ensure_dir(samples_dir / "val"), config.training.sample_count, use_probability, baseline_model)

    save_loss_curve(output_dir / "loss_curve.png", history["records"])
    dump_json(output_dir / "metrics.json", history)
    return history
