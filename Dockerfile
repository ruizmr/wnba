# ---------- Builder stage ----------
FROM continuumio/miniconda3:23.11.0-0 AS builder

WORKDIR /workspace

# Install Conda env first
COPY env.yml ./
RUN conda env create -f env.yml && conda clean -a -y
ENV PATH /opt/conda/envs/edge-env/bin:$PATH

# Copy source and run lint (does not fail build if hooks not installed)
COPY . /workspace
RUN conda run -n edge-env pre-commit run --all-files --show-diff-on-failure || true

# ---------- Runtime stage (distroless) ----------
FROM gcr.io/distroless/base-debian11 AS runtime

# Copy Python runtime & environment from builder
COPY --from=builder /opt/conda/envs/edge-env /opt/conda/envs/edge-env
COPY --from=builder /workspace /workspace

# Non-root user
RUN adduser --disabled-password --uid 1001 appuser
USER 1001:1001

ENV PATH /opt/conda/envs/edge-env/bin:$PATH
WORKDIR /workspace

EXPOSE 8000 8265 6379

ENTRYPOINT ["/opt/conda/envs/edge-env/bin/python", "-m", "serve.app"]