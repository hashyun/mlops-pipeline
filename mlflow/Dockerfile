FROM python:3.10-slim AS builder
LABEL authors="paradeigma"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/mlflow/.venv
WORKDIR /mlflow

RUN apt-get update && apt-get install -y gcc libc-dev && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

COPY mlflow/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import mlflow; print('mlflow LOADED!')"

FROM python:3.10-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/mlflow/.venv
WORKDIR /mlflow

COPY --from=builder /mlflow/.venv .venv
COPY --from=builder /mlflow/requirements.txt .

ENV PYTHONPATH='/usr/local/project'

ENV PATH="$VENV_PATH/bin:$PATH"

EXPOSE 5005

ENTRYPOINT ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5005 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --default-artifact-root \"$MLFLOW_DEFAULT_ARTIFACT_ROOT\""]
