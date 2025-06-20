########################
# ─── Build stage ───  #
########################
FROM python:3.10-slim AS builder

ENV \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VERSION=1.7.1

# System deps first (layer-cache friendly)
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry to craft a deterministic wheel-house
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /src
COPY pyproject.toml poetry.lock* ./
RUN poetry export --with main --without-hashes -f requirements.txt > /tmp/requirements.txt
RUN pip install --prefix=/venv -r /tmp/requirements.txt

# Pull the actual code last (so dep layers cache)
COPY . .
RUN pip install --prefix=/venv .

#########################
# ─── Runtime stage ─── #
#########################
FROM python:3.10-slim AS runtime
LABEL org.opencontainers.image.source="https://github.com/<org>/<repo>"
ENV PATH="/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user (uid 1001 keeps k8s happy)
RUN adduser --disabled-password --gecos "" --uid 1001 edge
USER edge
WORKDIR /app

# Copy venv & src from builder
COPY --from=builder /venv /venv
COPY --from=builder /src /app

EXPOSE 8000
ENTRYPOINT ["python", "-m", "wnba.serve.app"]