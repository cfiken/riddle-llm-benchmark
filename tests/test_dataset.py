import json
from pathlib import Path

import pytest

from riddle_benchmark.dataset.loader import DataLoader
from riddle_benchmark.dataset.schema import Riddle


@pytest.fixture
def mock_assets_dir(tmp_path):
    """Create a mock assets directory with metadata and images."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    images_dir = assets_dir / "images"
    images_dir.mkdir()

    # Create a dummy image
    image_path = images_dir / "test_riddle.png"
    image_path.touch()

    # Create metadata.jsonl
    metadata = [
        {
            "id": "test_001",
            "file_name": "images/test_riddle.png",
            "question": "What is this?",
            "answers": ["test", "TEST"],
            "hint": "It's a test.",
        }
    ]

    metadata_path = assets_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    return assets_dir


def test_riddle_schema_validation():
    """Test Riddle schema validation."""
    # Valid data
    riddle = Riddle(id="1", image_path=Path("img.png"), acceptable_answers=["a"])
    assert riddle.id == "1"
    assert riddle.question is None

    # Valid data with all fields
    riddle_full = Riddle(id="2", image_path=Path("img.png"), acceptable_answers=["a"], question="q", hint="h")
    assert riddle_full.question == "q"


def test_dataloader_load(mock_assets_dir):
    """Test loading data from the loader."""
    loader = DataLoader(data_dir=mock_assets_dir)
    riddles = loader.load()

    assert len(riddles) == 1
    riddle = riddles[0]

    assert riddle.id == "test_001"
    assert riddle.question == "What is this?"
    assert riddle.acceptable_answers == ["test", "TEST"]
    assert riddle.image_path.name == "test_riddle.png"
    assert riddle.image_path.exists()


def test_dataloader_missing_file(tmp_path):
    """Test loader behavior when metadata file is missing."""
    loader = DataLoader(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_dataloader_missing_image(tmp_path):
    """Test loader behavior when referenced image is missing."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    # Create metadata but NO image
    metadata = [{"id": "test_001", "file_name": "images/missing.png", "answers": ["a"]}]

    metadata_path = assets_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    loader = DataLoader(data_dir=assets_dir)

    with pytest.raises(ValueError, match="Image file not found"):
        loader.load()
