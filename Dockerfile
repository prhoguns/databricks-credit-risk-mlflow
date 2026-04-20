# Local stand-in for a Databricks runtime: Spark 3.4 + MLflow + the few libs the notebooks import.
FROM apache/spark-py:v3.4.0
USER root
RUN pip install --no-cache-dir mlflow==2.18.0 xlrd==2.0.1 pandas==2.2.3 scikit-learn==1.5.2 pyarrow==18.1.0 requests==2.32.3 tabulate==0.9.0
WORKDIR /app
ENV HOME=/tmp GIT_PYTHON_REFRESH=quiet
