FROM mcr.microsoft.com/playwright:v1.58.0-noble AS playwright

RUN apt-get update && apt-get install -y git curl libeccodes-dev && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# hadolint ignore=DL3029
ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

COPY . .
RUN rm -rf .venv uv.lock

RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Default port
ENV PORT=8000
EXPOSE ${PORT}

# Run the MCP server via uvicorn (HTTP transport) with graceful shutdown timeout
CMD ["sh", "-c", "uv run uvicorn app:app --port ${PORT} --host 0.0.0.0 --timeout-graceful-shutdown 1770"]
