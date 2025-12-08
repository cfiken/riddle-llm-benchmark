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
