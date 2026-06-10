from __future__ import annotations

import argparse

from saliency.config import load_config
from saliency.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/saliency/stage1_salicon_quick.toml",
        help="Path to the stage 1 config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.config)


if __name__ == "__main__":
    main()
