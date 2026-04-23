# Databricks notebook source
# MAGIC %md
# MAGIC # Config
# MAGIC Paths and the MLflow experiment. On Databricks `LAKE` is an ADLS path and MLflow is the workspace tracking
# MAGIC server; locally both fall back to folders so the notebooks run unchanged on plain Spark.

# COMMAND ----------

import os

import mlflow

LAKE = os.getenv("LAKE_ROOT", "abfss://gold@tordata.dfs.core.windows.net/credit_risk").rstrip("/")
TABLE_FORMAT = os.getenv("TABLE_FORMAT", "delta")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "/Shared/credit-default-risk")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_default_gbt")

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT)


def write(df, path, mode="overwrite"):
    w = df.write.format(TABLE_FORMAT).mode(mode)
    if TABLE_FORMAT == "delta":
        w = w.option("overwriteSchema", "true")
    w.save(path)


def read(path):
    return spark.read.format(TABLE_FORMAT).load(path)  # noqa: F821
