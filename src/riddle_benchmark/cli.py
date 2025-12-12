import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from riddle_benchmark.runner import BenchmarkRunner
from riddle_benchmark.utils import get_assets_path, get_logger, get_prompt_assets_path

logger = get_logger(__name__)


def main() -> None:
    # .env ファイルをロード
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the Riddle Benchmark.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The name of the model to benchmark.")
    parser.add_argument("--reason", action="store_true", help="Include reason in the response schema.")
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["0", "1", "2"],
        default="2",
        help="The prompt ID to use (0, 1 or 2). Refers to assets/prompts/{id}.txt. 0 means no prompt.",
    )
    parser.add_argument(
        "--extra-params",
        type=str,
        help='Additional model-specific parameters as JSON string (e.g., \'{"reasoning_effort": "high"}\').',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the results file. If not specified, saves to the current directory.",
    )

    args = parser.parse_args()

    # Parse extra_params if provided
    extra_params = None
    if args.extra_params:
        try:
            extra_params = json.loads(args.extra_params)
            if not isinstance(extra_params, dict):
                logger.error("--extra-params must be a valid JSON object")
                return
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --extra-params: {e}")
            return

    logger.info("ベンチマークを開始します...")
    logger.info(f"(Model: {args.model}, Reason: {args.reason}, Prompt: {args.prompt})")
    if extra_params:
        logger.info(f"Extra params: {extra_params}")

    # data_dir は assets ディレクトリを指定 (core.pyのヘルパーを利用)
    assets_dir = get_assets_path()

    prompt = None
    if args.prompt and args.prompt != "0":
        prompt_filename = f"{int(args.prompt):02d}.txt"
        prompt_path = get_prompt_assets_path() / prompt_filename
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"プロンプトファイルが見つかりません: {prompt_path}")

    runner = BenchmarkRunner(
        model_name=args.model,
        data_dir=assets_dir,
        use_reason=args.reason,
        prompt=prompt,
        extra_params=extra_params,
    )

    try:
        results = asyncio.run(runner.run())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Remove provider prefix if present (e.g. "gemini/gemini-1.5-pro" -> "gemini-1.5-pro")
        model_name_for_file = args.model.split("/")[-1]

        # 出力パスの決定
        default_filename = f"results_{model_name_for_file}_{timestamp}.json"
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / default_filename
        else:
            output_path = Path(default_filename)

        runner.save_report(output_path)
        logger.info(f"完了しました。結果は {output_path} に保存されました。")

        # 簡易サマリー表示
        summary = results["summary"]
        logger.info("--- 結果サマリー ---")
        logger.info(f"モデル: {summary['model']}")
        logger.info(f"正解数: {summary['correct_answers']} / {summary['total_questions']}")
        logger.info(f"正答率: {summary['accuracy']:.2%}")

    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    main()
