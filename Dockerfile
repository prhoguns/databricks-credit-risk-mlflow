# Local stand-in for a Databricks runtime: Spark 4.0 (the DBR 17 line) + MLflow 3 + the few libs the notebooks import.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
ENV HOME=/tmp GIT_PYTHON_REFRESH=quiet
