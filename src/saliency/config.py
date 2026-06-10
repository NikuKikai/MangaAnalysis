from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int


@dataclass(slots=True)
class DatasetConfig:
    name: str
    root: str
    input_size: int
    fixation_blur_sigma: float | None = None
    train_split: str | None = None
    val_split: str | None = None
    train_ratio: float | None = None
    val_ratio: float | None = None
    test_ratio: float | None = None
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None


@dataclass(slots=True)
class LoaderConfig:
    batch_size: int
    num_workers: int


@dataclass(slots=True)
class ModelConfig:
    encoder_pretrained: bool
    base_channels: int
    architecture_name: str = "v1"
    freeze_stages: int = 0


@dataclass(slots=True)
class OptimizerConfig:
    lr: float
    weight_decay: float


@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    device: str
    save_every: int
    sample_count: int
    eval_interval_steps: int = 100
    init_checkpoint: str | None = None
    compare_checkpoint: str | None = None
    compare_architecture_name: str | None = None


@dataclass(slots=True)
class LossConfig:
    kld_weight: float
    bce_weight: float
    mse_weight: float


@dataclass(slots=True)
class Config:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    loader: LoaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    loss: LossConfig


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path: str) -> Config:
    config_file = Path(config_path).resolve()
    with config_file.open("rb") as handle:
        raw = tomllib.load(handle)

    base_dir = config_file.parent.parent.parent
    dataset = raw["dataset"]
    experiment = raw["experiment"]
    training = raw["training"]

    dataset["root"] = _resolve_path(base_dir, dataset["root"])
    experiment["output_dir"] = _resolve_path(base_dir, experiment["output_dir"])
    if "init_checkpoint" in training:
        training["init_checkpoint"] = _resolve_path(base_dir, training["init_checkpoint"])
    if "compare_checkpoint" in training:
        training["compare_checkpoint"] = _resolve_path(base_dir, training["compare_checkpoint"])

    return Config(
        experiment=ExperimentConfig(**experiment),
        dataset=DatasetConfig(**dataset),
        loader=LoaderConfig(**raw["loader"]),
        model=ModelConfig(**raw["model"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        training=TrainingConfig(**training),
        loss=LossConfig(**raw["loss"]),
    )
