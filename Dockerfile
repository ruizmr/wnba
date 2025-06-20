# ---------- Base image ----------
FROM continuumio/miniconda3:23.11.0-0 AS base

# ---------- Build env ----------
WORKDIR /workspace

# Copy Conda spec first to leverage Docker layer caching
COPY env.yml ./
RUN conda env create -f env.yml && conda clean -a -y

# Make edge-env the default in every shell
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate edge-env" >> ~/.bashrc
ENV PATH /opt/conda/envs/edge-env/bin:$PATH

# ---------- Application layer ----------
COPY . /workspace

# Pre-commit lint on build (fails early if style broken)
RUN conda run -n edge-env pre-commit run --all-files --show-diff-on-failure || true

EXPOSE 8000 8265 6379

# ---------- Entrypoint ----------
# Start Ray head in the background then run the passed CMD (default: Serve app)
ENTRYPOINT ["/bin/bash", "-c", "ray start --head --port=6379 --dashboard-host 0.0.0.0 --dashboard-port 8265 && conda run --no-capture-output -n edge-env python -m serve.app"]

CMD []