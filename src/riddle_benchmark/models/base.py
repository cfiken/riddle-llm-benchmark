import base64
import json
import logging
from pathlib import Path
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from riddle_benchmark.dataset.schema import Riddle
from riddle_benchmark.utils import get_logger

logger = get_logger(__name__)

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

    @retry(
        stop=stop_after_attempt(3),  # 最大3回
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数バックオフ: 1秒、2秒、4秒、最大10秒
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # 最終的に失敗した場合は例外を再発生
    )
    async def solve(self, riddle: Riddle, response_schema: type[T], prompt: str | None = None) -> T:
        """
        Solve a riddle using the LLM with automatic retry on API errors.

        Args:
            riddle: The riddle to solve.
            response_schema: The Pydantic model to use for the response schema.
            prompt: Optional prompt to use for the request.

        Returns:
            The parsed response object (instance of response_schema).

        Raises:
            Various exceptions from litellm if all retry attempts fail.
        """
        messages = self._construct_messages(riddle, prompt)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Request] Model: {self.model_name}, Riddle ID: {riddle.id}")
            logger.debug(f"[Request] Messages: {self._format_messages_for_log(messages)}")
            logger.debug(f"[Request] Extra params: {self.kwargs}")

        response = await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            response_format=response_schema,
            **self.kwargs,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned empty content")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Response] Riddle ID: {riddle.id}")
            logger.debug(f"[Response] Content: {content}")

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

    def _format_messages_for_log(self, messages: list[dict[str, Any]]) -> str:
        """
        Format messages for logging, truncating image data.

        Args:
            messages: The messages list to format.

        Returns:
            A formatted string representation of the messages.
        """
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.get("role")}
            if "content" in msg:
                content = msg["content"]
                if isinstance(content, list):
                    formatted_content = []
                    for item in content:
                        if item.get("type") == "text":
                            formatted_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            # base64データを簡略化（最初の50文字と最後の10文字を表示）
                            if image_url.startswith("data:image"):
                                if len(image_url) > 100:
                                    truncated = f"{image_url[:50]}...{image_url[-10:]}"
                                else:
                                    truncated = image_url
                                formatted_content.append({"type": "image_url", "image_url": {"url": truncated}})
                            else:
                                formatted_content.append(item)
                    formatted_msg["content"] = formatted_content
                else:
                    formatted_msg["content"] = content
            formatted_messages.append(formatted_msg)
        return json.dumps(formatted_messages, indent=2, ensure_ascii=False)
