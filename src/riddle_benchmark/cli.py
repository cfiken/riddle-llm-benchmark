import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from riddle_benchmark.core import get_assets_path
from riddle_benchmark.runner import BenchmarkRunner


def main() -> None:
    # .env ファイルをロード
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the Riddle Benchmark.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The name of the model to benchmark.")
    parser.add_argument("--reasoning", action="store_true", help="Include reasoning in the model response.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print(".env ファイルに OPENAI_API_KEY を設定するか、環境変数をエクスポートしてください。")
        return

    print(f"ベンチマークを開始します... (Model: {args.model}, Reasoning: {args.reasoning})")

    # data_dir は assets ディレクトリを指定 (core.pyのヘルパーを利用)
    assets_dir = get_assets_path()

    runner = BenchmarkRunner(model_name=args.model, data_dir=assets_dir, use_reasoning=args.reasoning)

    try:
        results = runner.run()
        output_path = Path("benchmark_results.json")
        runner.save_report(output_path)
        print(f"\n完了しました。結果は {output_path} に保存されました。")

        # 簡易サマリー表示
        summary = results["summary"]
        print("\n--- 結果サマリー ---")
        print(f"モデル: {summary['model']}")
        print(f"正解数: {summary['correct_answers']} / {summary['total_questions']}")
        print(f"正答率: {summary['accuracy']:.2%}")

    except Exception as e:
        print(f"\n実行中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
