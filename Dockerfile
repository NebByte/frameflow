# The agent and the converter, in one image.
#
# CPU only, and deliberately so. The whole argument of this project is that most
# of a 270-degree side wall is already in the footage and can be recovered
# without a GPU or a generative model; the CUDA path is opt-in and lives in
# requirements-gpu.txt. That keeps this image small enough to cold-start on
# Cloud Run instead of dragging a gigabyte of CUDA wheels nobody will use.

FROM python:3.11-slim

# ffmpeg is not optional. OpenCV writes mp4v, which no browser will play, so
# every delivered file is transcoded to H.264 afterwards -- without ffmpeg the
# converter still runs and produces a film you cannot watch.
#
# curl is for installing uv, which is how the agent launches the official
# ClickHouse MCP server (`uvx mcp-clickhouse`) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv, for `uvx mcp-clickhouse`. Placed on PATH for every user.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, so edits to the source do not re-resolve the world.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]" httpx

COPY . .

# Pre-fetch the ClickHouse MCP server so the first ledger question in the demo
# does not stall behind a cold download.
RUN uvx --help >/dev/null 2>&1 || true

# Vertex AI rather than an API key: the service runs as a service account and
# picks up credentials from the metadata server, so no secret is baked in.
ENV GOOGLE_GENAI_USE_VERTEXAI=TRUE \
    GOOGLE_CLOUD_LOCATION=us-central1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080
CMD ["python", "server.py"]
