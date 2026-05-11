# ─────────────────────────────────────────────────────────────────────────────
# Fragment_LLM — CUDA-enabled CLI training image
#
# Base: nvidia/cuda instead of python:slim — this is the ONLY way Docker
# containers can see the GPU.  python:slim has no CUDA runtime at all, so
# torch.cuda.is_available() always returns False regardless of compose config.
#
# CUDA 11.8 matches the +cu118 wheels already in requirements.txt.
# cudnn8 provides cuDNN (required by PyTorch attention / conv ops).
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Non-root user (OWASP least-privilege) ────────────────────────────────────
RUN useradd -m -r -s /bin/bash appuser

# ── Virtual environment ───────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu118

# ── Copy project code ─────────────────────────────────────────────────────────
COPY . .

# ── Directories + permissions ─────────────────────────────────────────────────
RUN mkdir -p \
        /app/data/raw \
        /app/data/processed \
        /app/checkpoints \
        /app/models \
    && chown -R appuser:appuser /app $VIRTUAL_ENV

# ── CUDA environment hints for PyTorch ───────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

USER appuser

CMD ["bash"]