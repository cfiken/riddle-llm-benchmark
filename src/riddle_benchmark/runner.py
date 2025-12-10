import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from riddle_benchmark.dataset.loader import DataLoader
from riddle_benchmark.dataset.schema import Riddle
from riddle_benchmark.evaluation.evaluator import Evaluator
from riddle_benchmark.models.base import Model
from riddle_benchmark.models.schemas import SimpleResponse, ThinkingResponse
from riddle_benchmark.utils import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    Runner for the Riddle Benchmark.
    """

    def __init__(
        self,
        model_name: str,
        data_dir: Path | None = None,
        use_reason: bool = False,
        prompt: str | None = None,
        extra_params: dict[str, Any] | None = None,
        **model_kwargs: Any,
    ):
        """
        Initialize the benchmark runner.

        Args:
            model_name: Name of the model to benchmark.
            data_dir: Path to the dataset directory.
            use_reason: Whether to include reason in the response schema.
            prompt: Prompt to use for the model.
            extra_params: Additional model-specific parameters (e.g., reasoning_effort for OpenAI).
            **model_kwargs: Additional arguments for the model.
        """
        self.model_name = model_name
        self.use_reason = use_reason
        self.prompt = prompt
        self.extra_params = extra_params

        # Merge extra_params into model_kwargs
        merged_kwargs = {**model_kwargs}
        if extra_params:
            merged_kwargs.update(extra_params)

        self.model = Model(model_name, **merged_kwargs)
        self.loader = DataLoader(data_dir)
        self.results: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    async def run(self, concurrency: int = 5) -> dict[str, Any]:
        """
        Run the benchmark asynchronously.

        Args:
            concurrency: The maximum number of concurrent requests.

        Returns:
            A dictionary containing the summary and detailed results.
        """
        riddles = self.loader.load()
        correct_count = 0
        total_count = len(riddles)

        logger.info(f"Starting benchmark for model: {self.model_name}")
        logger.info(f"Total riddles: {total_count}")
        logger.info(f"Concurrency: {concurrency}")

        schema: type[ThinkingResponse] | type[SimpleResponse] = ThinkingResponse if self.use_reason else SimpleResponse

        semaphore = asyncio.Semaphore(concurrency)

        async def process_riddle(riddle: Riddle) -> dict[str, Any]:
            async with semaphore:
                try:
                    # Solve
                    prediction_obj = await self.model.solve(riddle, response_schema=schema, prompt=self.prompt)

                    raw_prediction = prediction_obj.answer
                    reason = getattr(prediction_obj, "reason", None)

                    # Evaluate
                    is_correct = Evaluator.evaluate(raw_prediction, riddle)

                    return {
                        "riddle_id": riddle.id,
                        "question": riddle.question,
                        "prediction": raw_prediction,
                        "reason": reason,
                        "normalized_prediction": Evaluator.normalize(raw_prediction),
                        "acceptable_answers": riddle.acceptable_answers,
                        "is_correct": is_correct,
                    }
                except Exception as e:
                    logger.error(f"Error solving riddle {riddle.id}: {e}", exc_info=True)
                    return {"riddle_id": riddle.id, "error": str(e), "is_correct": False}

        tasks = [process_riddle(riddle) for riddle in riddles]

        # Use tqdm with as_completed to show progress
        self.results = []
        for future in tqdm(asyncio.as_completed(tasks), total=total_count, desc="Solving riddles"):
            result = await future
            self.results.append(result)
            if result.get("is_correct"):
                correct_count += 1

        # Sort results by riddle_id for consistency
        self.results.sort(key=lambda x: x["riddle_id"])

        accuracy = correct_count / total_count if total_count > 0 else 0

        self.summary = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "use_reason": self.use_reason,
                "prompt": self.prompt,
                "extra_params": self.extra_params,
                "model_kwargs": self.model.kwargs,
            },
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": accuracy,
        }

        return {"summary": self.summary, "details": self.results}

    def save_report(self, output_path: Path) -> None:
        """
        Save the benchmark report to a JSON file.

        Args:
            output_path: Path to save the JSON report.
        """
        report = {"summary": self.summary, "details": self.results}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")
