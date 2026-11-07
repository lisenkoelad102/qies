FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/qies_control

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install torch==2.11.* --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["qies-control"]
CMD ["--help"]
