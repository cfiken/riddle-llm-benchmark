import os
from pathlib import Path

from dotenv import load_dotenv

from riddle_benchmark.core import get_assets_path
from riddle_benchmark.runner import BenchmarkRunner


def main() -> None:
    # .env ファイルをロード
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
        print(".env ファイルに OPENAI_API_KEY を設定するか、環境変数をエクスポートしてください。")
        return

    print("ベンチマークを開始します...")

    # モデル名は gpt-4o を指定
    # data_dir は assets ディレクトリを指定 (core.pyのヘルパーを利用)
    assets_dir = get_assets_path()

    runner = BenchmarkRunner(model_name="gpt-4o", data_dir=assets_dir)

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
