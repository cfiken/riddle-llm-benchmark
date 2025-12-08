import json
from pathlib import Path
from typing import List, Generator
from riddle_benchmark.dataset.schema import Riddle
from riddle_benchmark.core import get_image_assets_path, get_assets_path

class DataLoader:
    """
    Loader for riddle datasets.

    Assumes a directory structure compatible with Hugging Face ImageFolder:
    assets/
      metadata.jsonl  # Contains metadata for each riddle
      images/         # Contains image files
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the assets directory. If None, uses the default assets path.
        """
        self.data_dir = data_dir or get_assets_path()
        self.metadata_path = self.data_dir / "metadata.jsonl"
        self.images_dir = self.data_dir / "images"

    def load(self) -> List[Riddle]:
        """
        Load all riddles from the dataset.

        Returns:
            List of Riddle objects.
        """
        return list(self.iter_load())

    def iter_load(self) -> Generator[Riddle, None, None]:
        """
        Iteratively load riddles from the dataset.

        Yields:
            Riddle objects.

        Raises:
            FileNotFoundError: If metadata.jsonl is not found.
            ValueError: If an image file specified in metadata is missing.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Handle HF ImageFolder format where image path might be relative to data_dir
                # "file_name": "images/riddle_001.png" -> we need full path
                image_rel_path = data.get("file_name")
                if not image_rel_path:
                     raise ValueError(f"Missing 'file_name' in metadata: {data}")

                full_image_path = self.data_dir / image_rel_path

                if not full_image_path.exists():
                     raise ValueError(f"Image file not found: {full_image_path}")

                # Map JSON fields to Riddle schema
                # Expecting metadata to contain: id, question, answers, etc.
                # 'answers' in JSONL -> 'acceptable_answers' in Schema
                yield Riddle(
                    id=data["id"],
                    image_path=full_image_path,
                    acceptable_answers=data["answers"], # Mapping 'answers' to 'acceptable_answers'
                    question=data.get("question"),
                    hint=data.get("hint")
                )
