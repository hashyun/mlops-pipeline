FROM python:3.10-slim AS base
LABEL authors="paradeigma"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AIRFLOW_HOME=/usr/local/airflow
WORKDIR $AIRFLOW_HOME

RUN apt-get update && \
    apt-get install -y gcc libc-dev vim curl && \
    rm -rf /var/lib/apt/lists/*

FROM base AS builder

RUN python -m venv .venv
ENV PATH="$AIRFLOW_HOME/.venv/bin:$PATH"

COPY airflow/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import flask_session; print('flask_session LOADED!')"

FROM base AS runner

COPY --from=builder $AIRFLOW_HOME/.venv .venv
ENV PATH="$AIRFLOW_HOME/.venv/bin:$PATH"

ENV PYTHONPATH='/usr/local/project:/usr/local/project/airflow/dags'

WORKDIR /usr/local/airflow
COPY airflow/dags/ ./dags/
COPY airflow/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
