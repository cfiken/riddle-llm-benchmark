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
    # Use a dict to map riddle_id to response to handle async completion order
    responses = {
        "1": SimpleResponse(answer="a1"),
        "2": SimpleResponse(answer="wrong"),
    }

    async def mock_solve(riddle, *args, **kwargs):
        # Get riddle_id from the riddle object
        riddle_id = riddle.id if hasattr(riddle, "id") else ""
        return responses.get(riddle_id, SimpleResponse(answer="wrong"))

    mock_model.solve = mock_solve
    mock_model.kwargs = {"temperature": 0.7}  # Mock kwargs for summary

    # Mock evaluate to return True if prediction matches acceptable_answers
    # Note: evaluate is a classmethod, so when called as Evaluator.evaluate(prediction, riddle),
    # Python automatically passes cls as the first argument, so we get (cls, prediction, riddle)
    def mock_evaluate(*args, **kwargs):
        # When called as Evaluator.evaluate(prediction, riddle), args is (cls, prediction, riddle)
        # But when mocked, the mock might not pass cls, so handle both cases
        if len(args) == 2:
            prediction, riddle = args
        else:
            cls, prediction, riddle = args
        return prediction in riddle.acceptable_answers

    # Use side_effect to properly handle the classmethod call
    mock_evaluator.evaluate.side_effect = mock_evaluate
    # normalize is a staticmethod
    mock_evaluator.normalize = lambda x: x  # Identity for test

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

    details = results["details"]
    assert len(details) == 2

    # Results are sorted by riddle_id, so find by id
    detail_1 = next(d for d in details if d["riddle_id"] == "1")
    detail_2 = next(d for d in details if d["riddle_id"] == "2")

    assert detail_1["is_correct"] is True
    assert detail_2["is_correct"] is False

    # Verify summary matches details
    assert summary["correct_answers"] == 1
    assert summary["accuracy"] == 0.5
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
