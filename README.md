# Riddle LLM Benchmark

A Python library for benchmarking LLMs on riddle-solving tasks, using [uv](https://github.com/astral-sh/uv).

## Features

- Modern Python packaging with `pyproject.toml`
- Dependency management with `uv`
- Docker-based development environment
- GitHub Actions CI

## Installation

```bash
uv pip install riddle-llm-benchmark
```

## Development

### Requirements

- Docker
- uv (optional, for local management)

### Setup

```bash
# Start the development container
docker compose up -d

# Run tests inside the container
docker compose exec dev pytest
```

### Local Development (without Docker)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```

## Result (2025/12)

- openai
    - openai/gpt-4o
    - openai/gpt-5.1-2025-11-13
    - openai/gpt-5-2025-08-07
    - openai/gpt-5-mini-2025-08-07
    - openai/gpt-5-nano-2025-08-07
- gemini
    - gemini/gemini-3-pro-preview
    - gemini/gemini-2.5-pro
    - gemini/gemini-2.5-flash
    - gemini/gemini-2.5-flash-lite
- bedrock
    - bedrock/anthropic.claude-opus-4-5-20251101-v1:0
    - bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
    - bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
