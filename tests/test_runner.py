import json
from pathlib import Path
from unittest.mock import patch

import pytest

from riddle_benchmark.dataset.schema import Riddle
from riddle_benchmark.models.schemas import SimpleResponse
from riddle_benchmark.runner import BenchmarkRunner


@pytest.fixture
def mock_riddles():
    return [
        Riddle(id="1", image_path=Path("img1.png"), question="q1", acceptable_answers=["a1"]),
        Riddle(id="2", image_path=Path("img2.png"), question="q2", acceptable_answers=["a2"]),
    ]


@patch("riddle_benchmark.runner.DataLoader")
@patch("riddle_benchmark.runner.Model")
@patch("riddle_benchmark.runner.Evaluator")
@pytest.mark.asyncio
async def test_runner_run(mock_evaluator, mock_model_class, mock_loader_class, mock_riddles):
    # Setup mocks
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = mock_riddles

    mock_model = mock_model_class.return_value
    # First riddle correct, second incorrect
    responses = [SimpleResponse(answer="a1"), SimpleResponse(answer="wrong")]

    async def mock_solve(*args, **kwargs):
        return responses.pop(0) if responses else SimpleResponse(answer="wrong")

    mock_model.solve = mock_solve
    mock_model.kwargs = {"temperature": 0.7}  # Mock kwargs for summary

    mock_evaluator.evaluate.side_effect = [True, False]
    mock_evaluator.normalize.side_effect = lambda x: x  # Identity for test

    # Initialize runner
    runner = BenchmarkRunner(model_name="test-model", temperature=0.7, prompt="SysPrompt")

    # Run benchmark
    results = await runner.run()

    # Verify model initialization
    mock_model_class.assert_called_with("test-model", temperature=0.7)

    # Verify results structure
    assert "summary" in results
    assert "details" in results

    summary = results["summary"]
    assert summary["model"] == "test-model"
    assert summary["total_questions"] == 2
    assert summary["correct_answers"] == 1
    assert summary["accuracy"] == 0.5

    details = results["details"]
    assert len(details) == 2
    assert details[0]["is_correct"] is True
    assert details[1]["is_correct"] is False
    assert details[0]["riddle_id"] == "1"
    assert details[1]["riddle_id"] == "2"
    # difficulty/category checks removed


@patch("riddle_benchmark.runner.DataLoader")
@patch("riddle_benchmark.runner.Model")
@pytest.mark.asyncio
async def test_runner_error_handling(mock_model_class, mock_loader_class, mock_riddles):
    # Setup mocks
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = [mock_riddles[0]]

    mock_model = mock_model_class.return_value

    async def mock_solve_error(*args, **kwargs):
        raise Exception("API Error")

    mock_model.solve = mock_solve_error
    mock_model.kwargs = {}

    runner = BenchmarkRunner(model_name="test-model")
    results = await runner.run()

    details = results["details"]
    assert len(details) == 1
    assert details[0]["is_correct"] is False
    assert "error" in details[0]
    assert details[0]["error"] == "API Error"

    summary = results["summary"]
    assert summary["correct_answers"] == 0
    assert summary["accuracy"] == 0.0


def test_save_report(tmp_path):
    runner = BenchmarkRunner(model_name="test")
    # Manually populate results/summary to test save
    runner.summary = {"test": "summary"}
    runner.results = [{"test": "result"}]

    output_path = tmp_path / "report.json"
    runner.save_report(output_path)

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
        assert data["summary"] == {"test": "summary"}
        assert data["details"] == [{"test": "result"}]
