"""Dump the MLflow experiment and registry to docs/mlflow_runs.md so the results are visible without the UI."""
import os

import mlflow
import pandas as pd

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:////app/lake/mlflow.db"))
runs = mlflow.search_runs(experiment_names=["credit-default-risk"], filter_string="tags.mlflow.runName != 'batch_score'")
cols = ["tags.mlflow.runName", "params.algorithm", "params.maxDepth", "params.maxIter", "params.regParam", "params.elasticNetParam", "metrics.test_auc", "metrics.test_pr_auc", "metrics.test_ks", "metrics.cv_best_auc"]
cols = [c for c in cols if c in runs.columns]
t = runs[cols].rename(columns=lambda c: c.split(".", 1)[1])
t = t.rename(columns={"mlflow.runName": "run"})
num = t.select_dtypes("number").columns
t[num] = t[num].astype(float).round(4)
client = mlflow.tracking.MlflowClient()
champion = client.get_model_version_by_alias("credit_default_gbt", "champion")
versions = [(v.name, v.version, ["champion"] if v.version == champion.version else [], v.tags.get("test_auc"), v.run_id[:8]) for v in client.search_model_versions("name='credit_default_gbt'")]
lines = ["# MLflow experiment: credit-default-risk", "", "Runs (from `mlflow.search_runs`):", "", t.to_markdown(index=False), "", "Model registry:", "", "| model | version | aliases | test_auc tag | run |", "|---|---:|---|---:|---|"]
lines += [f"| {n} | {v} | {', '.join(a)} | {auc} | {r} |" for n, v, a, auc, r in versions]
open("docs/mlflow_runs.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
