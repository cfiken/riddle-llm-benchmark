# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the project files
# First copy only the files needed for installation to cache dependencies
COPY pyproject.toml uv.lock ./

# Install the project's dependencies
# --frozen: use the lockfile exactly as is
# --no-install-project: don't install the project itself yet (useful for caching)
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the project
COPY . .

# Install the project and dev dependencies
RUN uv sync --frozen

# Place the virtual environment in the path
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python"]

