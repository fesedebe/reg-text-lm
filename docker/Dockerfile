
# Build: docker build -f docker/Dockerfile -t legal-simplification .
# Requires: NVIDIA GPU + nvidia-container-toolkit on host
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

COPY docker/requirements-lock.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements-lock.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pyproject.toml README.md .

EXPOSE 8000
CMD ["bash"]
