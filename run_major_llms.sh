#!/bin/bash

# デフォルト値
DEFAULT_REASON=false
DEFAULT_PROMPT=0
DEFAULT_REASONING_EFFORT="none"  # デフォルトのreasoning_effort値
DEFAULT_OUTPUT_DIR=""  # デフォルトは空（カレントディレクトリ）

reason="${1:-$DEFAULT_REASON}"
prompt="${2:-$DEFAULT_PROMPT}"
reasoning_effort="${3:-$DEFAULT_REASONING_EFFORT}"  # 第3引数でreasoning_effortを指定可能
output_dir="${4:-$DEFAULT_OUTPUT_DIR}"  # 第4引数で出力ディレクトリを指定可能

echo "Settings: Reason=$reason, Prompt=$prompt, ReasoningEffort=$reasoning_effort, OutputDir=${output_dir:-.}"

# モデル定義
models=(
    "openai/gpt-4o"
    "openai/gpt-5.1-2025-11-13"
    "openai/gpt-5-2025-08-07"
    "openai/gpt-5-mini-2025-08-07"
    "openai/gpt-5-nano-2025-08-07"
    "gemini/gemini-3-pro-preview"
    "gemini/gemini-2.5-pro"
    "gemini/gemini-2.5-flash"
    "gemini/gemini-2.5-flash-lite"
    "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
    "bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0"
    "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# モデルごとにreasoning_effortをサポートしているかチェックする関数
supports_reasoning_effort() {
    local model="$1"
    # GPT-4oはreasoning_effortをサポートしていない
    if [[ "$model" == *"gpt-4o"* ]]; then
        return 1
    fi
    # その他のモデルはサポートしている（GPT-5系、Gemini系、Claude系）
    return 0
}

# モデルごとに実行
for model in "${models[@]}"; do
    # 基本コマンド構築
    base_cmd="uv run riddle-benchmark --model \"$model\" --prompt \"$prompt\""

    # --reasonフラグの追加
    if [ "$reason" = true ]; then
        base_cmd="$base_cmd --reason"
    fi

    # reasoning_effortの設定（サポートしている場合かつnone以外の場合）
    if supports_reasoning_effort "$model" && [ "$reasoning_effort" != "none" ]; then
        extra_params_json="{\"reasoning_effort\": \"$reasoning_effort\"}"
        base_cmd="$base_cmd --extra-params '$extra_params_json'"
    elif [ "$reasoning_effort" != "none" ]; then
        echo "Warning: $model does not support reasoning_effort, skipping reasoning parameter"
    fi

    # 出力ディレクトリの設定
    if [ -n "$output_dir" ]; then
        base_cmd="$base_cmd --output-dir \"$output_dir\""
    fi

    echo "Running: $base_cmd"
    eval $base_cmd
done
