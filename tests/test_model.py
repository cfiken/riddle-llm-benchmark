import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from riddle_benchmark.dataset.schema import Riddle
from riddle_benchmark.models.base import Model
from riddle_benchmark.models.schemas import SimpleResponse


@pytest.fixture
def mock_riddle():
    return Riddle(
        id="test_001",
        image_path=Path("test_image.jpg"),
        question="What is this?",
        acceptable_answers=["test"],
        hint="It's a test.",
    )


@patch("riddle_benchmark.models.base.litellm.completion")
@patch("builtins.open", new_callable=MagicMock)
def test_model_solve(mock_open, mock_completion, mock_riddle):
    # Setup mock for file opening (for image encoding)
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake_image_content"
    mock_open.return_value.__enter__.return_value = mock_file

    # Setup mock for litellm response
    mock_response = MagicMock()
    # Content should be a JSON string matching the schema
    response_content = json.dumps({"answer": "test answer"})
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_completion.return_value = mock_response

    # Initialize model
    model = Model(model_name="gpt-4o", temperature=0.5)

    # Run solve
    result = model.solve(mock_riddle, SimpleResponse)

    # Verify response
    assert isinstance(result, SimpleResponse)
    assert result.answer == "test answer"

    # Verify litellm.completion was called correctly
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args.kwargs["model"] == "gpt-4o"
    assert call_args.kwargs["temperature"] == 0.5
    # response_format should be passed
    assert call_args.kwargs["response_format"] == SimpleResponse

    messages = call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert "Question: What is this?" in content[0]["text"]
    assert "Hint: It's a test." in content[0]["text"]

    assert content[1]["type"] == "image_url"
    # base64 encoded "fake_image_content" is "ZmFrZV9pbWFnZV9jb250ZW50"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZV9pbWFnZV9jb250ZW50"


@patch("riddle_benchmark.models.base.litellm.completion")
@patch("builtins.open", new_callable=MagicMock)
def test_model_solve_with_prompt(mock_open, mock_completion, mock_riddle):
    # Setup mocks
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake_image_content"
    mock_open.return_value.__enter__.return_value = mock_file

    mock_response = MagicMock()
    response_content = json.dumps({"answer": "test answer"})
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_completion.return_value = mock_response

    # Initialize model with prompt
    prompt = "You are a riddle solver."
    model = Model(model_name="gpt-4o")

    # Run solve
    model.solve(mock_riddle, SimpleResponse, prompt=prompt)

    # Verify messages structure
    messages = mock_completion.call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert prompt in content[0]["text"]
    assert "Question:" in content[0]["text"]


@patch("riddle_benchmark.models.base.litellm.completion")
@patch("builtins.open", new_callable=MagicMock)
def test_model_solve_no_hint(mock_open, mock_completion, mock_riddle):
    # Modify riddle to have no hint
    mock_riddle.hint = None

    # Setup mocks
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake_image_content"
    mock_open.return_value.__enter__.return_value = mock_file

    mock_response = MagicMock()
    response_content = json.dumps({"answer": "answer"})
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_completion.return_value = mock_response

    model = Model(model_name="gpt-4o")
    model.solve(mock_riddle, SimpleResponse)

    # Verify messages structure when no hint is present
    messages = mock_completion.call_args.kwargs["messages"]
    content = messages[0]["content"]
    assert "Hint:" not in content[0]["text"]


@patch("riddle_benchmark.models.base.litellm.completion")
@patch("builtins.open", new_callable=MagicMock)
def test_model_solve_no_question(mock_open, mock_completion, mock_riddle):
    # Modify riddle to have no question
    mock_riddle.question = None

    # Setup mocks
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake_image_content"
    mock_open.return_value.__enter__.return_value = mock_file

    mock_response = MagicMock()
    response_content = json.dumps({"answer": "answer"})
    mock_response.choices = [MagicMock(message=MagicMock(content=response_content))]
    mock_completion.return_value = mock_response

    model = Model(model_name="gpt-4o")
    model.solve(mock_riddle, SimpleResponse)

    # Verify messages structure when no question is present
    messages = mock_completion.call_args.kwargs["messages"]
    content = messages[0]["content"]

    # Should contain only image_url if no prompt, no question, no hint (hint is present here though)
    # But hint IS present in mock_riddle, so text part should exist for Hint
    text_part = next((item for item in content if item["type"] == "text"), None)
    assert text_part is not None
    assert "Question:" not in text_part["text"]
    assert "Hint: It's a test." in text_part["text"]
