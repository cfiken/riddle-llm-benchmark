#!/bin/bash

# デフォルト値
DEFAULT_REASONING=false
DEFAULT_PROMPT=0

reasoning="${1:-$DEFAULT_REASONING}"
prompt="${2:-$DEFAULT_PROMPT}"

echo "Settings: Reasoning=$reasoning, Prompt=$prompt"

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



if [ "$reasoning" = true ]; then
    cmd="uv run riddle-benchmark --model \"\$model\" --reasoning --prompt \"\$prompt\""
else
    cmd="uv run riddle-benchmark --model \"\$model\" --prompt \"\$prompt\""
fi

for model in "${models[@]}"; do
    eval $cmd
done
