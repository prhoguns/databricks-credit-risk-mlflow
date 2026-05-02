"""Run the three notebooks on local Spark with a file-based MLflow store. Same code that runs on Databricks."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

NB = Path(__file__).resolve().parents[1] / "notebooks"
ORDER = ["01_ingest.py", "02_train.py", "03_batch_score.py"]


def source(name: str) -> str:
    out = []
    for line in (NB / name).read_text().splitlines():
        if line.startswith("# MAGIC %run"):
            out.append((NB / "00_config.py").read_text())
        elif line.startswith(("# MAGIC", "# COMMAND", "# Databricks notebook source")):
            continue
        else:
            out.append(line)
    return "\n".join(out)


def main() -> None:
    root = Path(os.getenv("WORK_DIR", "/app/lake")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LAKE_ROOT", str(root))
    os.environ.setdefault("TABLE_FORMAT", "parquet")
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlflow.db")
    os.environ.setdefault("MLFLOW_EXPERIMENT", "credit-default-risk")
    if len(sys.argv) > 1:
        os.environ["CREDIT_ZIP"] = sys.argv[1]

    spark = SparkSession.builder.master("local[*]").appName("credit-risk-local").config("spark.sql.shuffle.partitions", "8").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    for nb in ORDER:
        print(f"\n=== {nb}")
        exec(compile(source(nb), nb, "exec"), {"spark": spark, "os": os, "__name__": "__notebook__"})
    print(f"\nMLflow store: {os.environ['MLFLOW_TRACKING_URI']}  (mlflow ui --backend-store-uri ... to browse)")


if __name__ == "__main__":
    main()
