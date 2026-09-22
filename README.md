# Credit Default Risk on Databricks with MLflow

_Portfolio sprint timeline: January–September 2026. Reported results retain their actual run dates._

Spark MLlib credit-risk model with the full MLflow lifecycle — experiment tracking, cross-validated
tuning, feature importances, model registry with an alias, and a batch-scoring job that loads the
model by alias rather than by file. Written as Databricks notebooks; verified end to end on plain
Spark in Docker, so every number below comes from a real run.

| | |
|---|---|
| Data | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350): 30,000 customers, 23 attributes, 22.1% default rate (CC BY 4.0) |
| Notebooks | [`01_ingest`](notebooks/01_ingest.py) → bronze · [`02_train`](notebooks/02_train.py) → MLflow runs + registry · [`03_batch_score`](notebooks/03_batch_score.py) → gold scores |
| Models | Logistic regression (interpretable baseline) vs gradient-boosted trees (challenger), 3-fold CV over a small grid |
| Tracking | Params, test AUC / PR-AUC / **KS statistic**, feature importances, Spark model artifact, registry alias `@champion` |
| Evidence | [`docs/mlflow_runs.md`](docs/mlflow_runs.md) — exported from the MLflow store after the run |

## Results

| run | tuned params | test AUC | PR-AUC | KS | CV AUC |
|---|---|---:|---:|---:|---:|
| gradient_boosted_trees (`@champion`) | maxDepth=5, maxIter=40 | **0.783** | 0.554 | **0.426** | 0.780 |
| logistic_regression | regParam=0.01, elasticNet=0 | 0.756 | 0.507 | 0.407 | 0.756 |

Batch scoring puts customers in three bands; the bands are well calibrated against actual outcomes:

| risk band | customers | mean predicted p | actual default rate |
|---|---:|---:|---:|
| low (< 0.25) | 22,081 | 0.131 | 0.114 |
| medium | 4,338 | 0.352 | 0.364 |
| high (≥ 0.50) | 3,581 | 0.687 | 0.706 |

For context, published results on this dataset cluster around AUC 0.77–0.78; the model is at the
ceiling the features allow, not above it. KS is reported because it is the number credit-risk teams
quote; the top-decile lift is what a collections team would act on.

## What each piece is for

- **Feature engineering in PySpark** ([`02_train`](notebooks/02_train.py)): utilisation, payment ratio,
  months late, max months late, bill trend — the derived features carry most of the importance, which
  is the point of doing them.
- **`mlflow.start_run` per candidate**: every run stores its params, metrics, `feature_importances.json`
  and the fitted `PipelineModel` via `mlflow.spark.log_model`, so any run can be reloaded and scored.
- **`mlflow.register_model` + alias `champion`**: the scoring notebook asks for
  `models:/credit_default_gbt@champion`. Promoting a new model is one alias change; no scoring code
  changes.
- **Batch score logs its own run**: the band counts and actual rates are metrics too, so a shift in the
  score distribution between runs is visible in the same UI.

## Run it locally (no Databricks needed)

```bash
git clone https://github.com/prhoguns/databricks-credit-risk-mlflow.git
cd databricks-credit-risk-mlflow
docker build -t credit-risk .
docker run --rm -v "$PWD":/app credit-risk /opt/spark/bin/spark-submit tests/run_local.py      # ~4 min, downloads the data
docker run --rm -v "$PWD":/app -p 5000:5000 credit-risk mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:////app/lake/mlflow.db
```

`tests/run_local.py` strips the Databricks `%run`/`# MAGIC` lines and executes the notebooks on a
local SparkSession with a SQLite-backed MLflow store. `TABLE_FORMAT=parquet` locally; Delta on Databricks.

## Run it on Databricks

1. Repos → add this repository. Open `notebooks/02_train`.
2. The workspace tracking server and Unity Catalog registry are used automatically (`MLFLOW_TRACKING_URI` unset).
3. Set `LAKE_ROOT` to an ADLS path (e.g. the gold container from
   [azure-toronto-data-platform](https://github.com/prhoguns/azure-toronto-data-platform)) or leave the default.
4. Run `01_ingest`, `02_train`, `03_batch_score` in order; then Models → `credit_default_gbt` → `@champion`.

## Next

- Champion/challenger promotion gated on a metric threshold, as a Databricks Job.
- SHAP values per customer for adverse-action reasons (a regulatory requirement for credit decisions).
- Drift monitoring: population stability index on the score distribution per scoring run.

## Acknowledgments

AI tools assisted with documentation and repository organization.
