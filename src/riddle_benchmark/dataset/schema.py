from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel

class Riddle(BaseModel):
    """
    Riddle data schema.

    Attributes:
        id: Unique identifier for the riddle
        image_path: Path to the image file
        acceptable_answers: List of acceptable answers
        question: The riddle question (optional)
        hint: Optional hint for the riddle
    """
    id: str
    image_path: Path
    acceptable_answers: List[str]
    question: Optional[str] = None
    hint: Optional[str] = None
