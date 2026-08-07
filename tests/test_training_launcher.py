import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "saigformer_training_launcher",
    REPOSITORY_ROOT / "train.py",
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LAUNCHER)


def _write_image_placeholders(root: Path, split: str, names):
    for branch in ("input", "target"):
        directory = root / "datasets" / "LOLv1" / split / branch
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / f"{name}.png").touch()


def test_lolv1_alias_resolves_full_rcsf_recipe():
    config = LAUNCHER.resolve_config("lolv1", REPOSITORY_ROOT)
    assert config.name == "CAME_SAIGFormer_lolv1_rcsf.yml"


def test_lolv1_dataset_validation_counts_paired_images(tmp_path):
    _write_image_placeholders(tmp_path, "Train", ("a", "b"))
    _write_image_placeholders(tmp_path, "Test", ("c",))

    assert LAUNCHER.validate_dataset("lolv1", tmp_path) == {
        "train": 2,
        "test": 1,
    }


def test_lolv1_dataset_validation_rejects_unpaired_names(tmp_path):
    _write_image_placeholders(tmp_path, "Train", ("a",))
    _write_image_placeholders(tmp_path, "Test", ("b",))
    (tmp_path / "datasets" / "LOLv1" / "Train" / "target" / "a.png").unlink()
    (tmp_path / "datasets" / "LOLv1" / "Train" / "target" / "other.png").touch()

    with pytest.raises(RuntimeError, match="not paired"):
        LAUNCHER.validate_dataset("lolv1", tmp_path)
