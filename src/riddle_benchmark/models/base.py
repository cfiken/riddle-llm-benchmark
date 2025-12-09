import base64
from pathlib import Path
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from riddle_benchmark.dataset.schema import Riddle

T = TypeVar("T", bound=BaseModel)


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

    def solve(self, riddle: Riddle, response_schema: type[T], prompt: str | None = None) -> T:
        """
        Solve a riddle using the LLM.

        Args:
            riddle: The riddle to solve.
            response_schema: The Pydantic model to use for the response schema.
            prompt: Optional prompt to use for the request.

        Returns:
            The parsed response object (instance of response_schema).
        """
        messages = self._construct_messages(riddle, prompt)

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            response_format=response_schema,
            **self.kwargs,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned empty content")

        return response_schema.model_validate_json(content)

    def _construct_messages(self, riddle: Riddle, prompt: str | None = None) -> list[dict[str, Any]]:
        """
        Construct the messages payload for LiteLLM.

        Args:
            riddle: The riddle object.
            prompt: Optional prompt to use for the request.

        Returns:
            A list of message dictionaries.
        """
        text_components = []
        if prompt:
            text_components.append(prompt)

        question_hint_components = []
        if riddle.question:
            question_hint_components.append(f"Question: {riddle.question}")
        if riddle.hint:
            question_hint_components.append(f"Hint: {riddle.hint}")

        if question_hint_components:
            text_components.append("\n".join(question_hint_components))

        final_text = "\n\n".join(text_components)

        # Construct the user content with both text and image
        content: list[dict[str, Any]] = []

        if final_text:
            content.append(
                {
                    "type": "text",
                    "text": final_text,
                }
            )

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(riddle.image_path)}"},
            }
        )

        messages = [{"role": "user", "content": content}]

        return messages

    def _encode_image(self, image_path: Path) -> str:
        """
        Encode an image file to base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
