import base64
from pathlib import Path
from typing import Any

import litellm

from riddle_benchmark.dataset.schema import Riddle


class Model:
    """
    A unified interface for LLMs using LiteLLM.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initialize the model wrapper.

        Args:
            model_name: The name of the model to use (e.g., "gpt-4o", "gemini-1.5-pro").
            **kwargs: Additional arguments to pass to litellm.completion.
        """
        self.model_name = model_name
        self.kwargs = kwargs

    def solve(self, riddle: Riddle) -> str:
        """
        Solve a riddle using the LLM.

        Args:
            riddle: The riddle to solve.

        Returns:
            The raw string output from the model.
        """
        messages = self._construct_messages(riddle)

        response = litellm.completion(model=self.model_name, messages=messages, **self.kwargs)

        return str(response.choices[0].message.content)

    def _construct_messages(self, riddle: Riddle) -> list[dict[str, Any]]:
        """
        Construct the messages payload for LiteLLM.

        Args:
            riddle: The riddle object.

        Returns:
            A list of message dictionaries.
        """
        # Default prompt text if question is missing
        question_text = riddle.question if riddle.question else "What does this image represent?"

        # Construct the user content with both text and image
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Question: {question_text}\n\nPlease provide only the answer word, nothing else.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(riddle.image_path)}"},
            },
        ]

        # Add hint if available
        if riddle.hint:
            content[0]["text"] += f"\nHint: {riddle.hint}"

        return [{"role": "user", "content": content}]

    def _encode_image(self, image_path: Path) -> str:
        """
        Encode an image file to base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
