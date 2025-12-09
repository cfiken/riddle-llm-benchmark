import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from riddle_benchmark.dataset.loader import DataLoader
from riddle_benchmark.evaluation.evaluator import Evaluator
from riddle_benchmark.models.base import Model
from riddle_benchmark.models.schemas import ReasoningResponse, SimpleResponse


class BenchmarkRunner:
    """
    Runner for the Riddle Benchmark.
    """

    def __init__(
        self,
        model_name: str,
        data_dir: Path | None = None,
        use_reasoning: bool = False,
        **model_kwargs: Any,
    ):
        """
        Initialize the benchmark runner.

        Args:
            model_name: Name of the model to benchmark.
            data_dir: Path to the dataset directory.
            use_reasoning: Whether to ask the model for reasoning.
            **model_kwargs: Additional arguments for the model.
        """
        self.model_name = model_name
        self.use_reasoning = use_reasoning
        self.model = Model(model_name, **model_kwargs)
        self.loader = DataLoader(data_dir)
        self.results: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """
        Run the benchmark.

        Returns:
            A dictionary containing the summary and detailed results.
        """
        riddles = self.loader.load()
        correct_count = 0
        total_count = len(riddles)

        print(f"Starting benchmark for model: {self.model_name}")
        print(f"Total riddles: {total_count}")

        schema: type[ReasoningResponse] | type[SimpleResponse] = (
            ReasoningResponse if self.use_reasoning else SimpleResponse
        )

        for riddle in tqdm(riddles, desc="Solving riddles"):
            try:
                # Solve
                prediction_obj = self.model.solve(riddle, response_schema=schema)

                raw_prediction = prediction_obj.answer
                reasoning = getattr(prediction_obj, "reasoning", None)

                # Evaluate
                is_correct = Evaluator.evaluate(raw_prediction, riddle)

                if is_correct:
                    correct_count += 1

                # Record result
                self.results.append(
                    {
                        "riddle_id": riddle.id,
                        "question": riddle.question,
                        "prediction": raw_prediction,
                        "reasoning": reasoning,
                        "normalized_prediction": Evaluator.normalize(raw_prediction),
                        "acceptable_answers": riddle.acceptable_answers,
                        "is_correct": is_correct,
                    }
                )

            except Exception as e:
                print(f"Error solving riddle {riddle.id}: {e}")
                self.results.append({"riddle_id": riddle.id, "error": str(e), "is_correct": False})

        accuracy = correct_count / total_count if total_count > 0 else 0

        self.summary = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
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
        print(f"Report saved to {output_path}")
