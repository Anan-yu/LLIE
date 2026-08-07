"""Short, repository-root training launcher.

Examples:
    python train.py lolv1
    python train.py lolv1 --gpu 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parent
TRAINING_CONFIGS: Dict[str, str] = {
    "lolv1": "Options/CAME_SAIGFormer_lolv1_rcsf.yml",
}
DATASET_DIRECTORIES: Dict[str, Tuple[str, ...]] = {
    "lolv1": (
        "datasets/LOLv1/Train/input",
        "datasets/LOLv1/Train/target",
        "datasets/LOLv1/Test/input",
        "datasets/LOLv1/Test/target",
    ),
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def resolve_config(dataset: str, repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Resolve the canonical paper configuration for a dataset alias."""
    try:
        relative_path = TRAINING_CONFIGS[dataset]
    except KeyError as error:
        supported = ", ".join(sorted(TRAINING_CONFIGS))
        raise ValueError(
            f"Unsupported dataset alias '{dataset}'. Supported aliases: {supported}."
        ) from error

    config_path = repository_root / relative_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    return config_path


def _image_stems(directory: Path) -> set[str]:
    return {
        item.stem
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_dataset(
    dataset: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> Dict[str, int]:
    """Fail early when a paired dataset is absent, empty, or misaligned."""
    directories = tuple(repository_root / item for item in DATASET_DIRECTORIES[dataset])
    missing = [str(path) for path in directories if not path.is_dir()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required LOLv1 directories are missing:\n"
            f"  - {formatted}\n"
            "Expected Train/Test directories with input and target subdirectories."
        )

    train_input, train_target, test_input, test_target = (
        _image_stems(path) for path in directories
    )
    pairs: Iterable[Tuple[str, set[str], set[str]]] = (
        ("Train", train_input, train_target),
        ("Test", test_input, test_target),
    )
    counts: Dict[str, int] = {}
    for split, inputs, targets in pairs:
        if not inputs or not targets:
            raise RuntimeError(f"{split} input/target directories must contain images.")
        if inputs != targets:
            missing_targets = sorted(inputs - targets)[:5]
            missing_inputs = sorted(targets - inputs)[:5]
            raise RuntimeError(
                f"{split} input/target filenames are not paired. "
                f"Missing targets: {missing_targets}; missing inputs: {missing_inputs}."
            )
        counts[split.lower()] = len(inputs)
    return counts


def build_command(config_path: Path) -> list[str]:
    """Use the active environment's Python for the actual trainer."""
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "basicsr" / "train.py"),
        "--opt",
        str(config_path),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command SAIGFormer training launcher."
    )
    parser.add_argument(
        "dataset",
        choices=sorted(TRAINING_CONFIGS),
        help="Dataset recipe to train. 'lolv1' selects the full RCSF method.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA device index. Defaults to CUDA_VISIBLE_DEVICES or GPU 0.",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip the paired directory and filename checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected command without starting training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_config(args.dataset)

    if not args.skip_data_check:
        counts = validate_dataset(args.dataset)
        print(
            f"Dataset check passed: {counts['train']} training pairs, "
            f"{counts['test']} test pairs.",
            flush=True,
        )

    environment = os.environ.copy()
    if args.gpu is None:
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    command = build_command(config_path)
    print(f"Training recipe: {config_path.relative_to(REPOSITORY_ROOT)}", flush=True)
    print(
        f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )
    print("Automatic resume is enabled when a training state exists.", flush=True)
    if args.dry_run:
        print("Command: " + " ".join(command), flush=True)
        return 0

    try:
        return subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
