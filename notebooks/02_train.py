# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Features and training with Spark MLlib, tracked in MLflow
# MAGIC Feature engineering in PySpark, two candidate models (logistic regression as the interpretable baseline,
# MAGIC gradient-boosted trees as the challenger), a small cross-validated grid, every run logged to MLflow with
# MAGIC params, metrics, feature importances and the model artifact. The best model is registered.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

df = read(f"{LAKE}/bronze/credit_default")

# COMMAND ----------

# MAGIC %md ## Feature engineering
# MAGIC Raw columns are six months of `pay_k` (months late), `bill_amtk` and `pay_amtk`. The derived features are what a
# MAGIC credit analyst would compute by hand: utilisation, how often the customer was late, and whether balances are growing.

# COMMAND ----------

months = range(1, 7)
feat = (
    df.withColumn("avg_bill", sum(F.col(f"bill_amt{m}") for m in months) / 6)
    .withColumn("avg_pay", sum(F.col(f"pay_amt{m}") for m in months) / 6)
    .withColumn("utilisation", F.col("avg_bill") / F.col("limit_bal"))
    .withColumn("pay_ratio", F.col("avg_pay") / F.greatest(F.col("avg_bill"), F.lit(1.0)))
    .withColumn("months_late", sum(F.when(F.col(f"pay_{m}") > 0, 1).otherwise(0) for m in months))
    .withColumn("max_months_late", F.greatest(*[F.col(f"pay_{m}") for m in months]))
    .withColumn("bill_trend", (F.col("bill_amt1") - F.col("bill_amt6")) / F.greatest(F.abs(F.col("bill_amt6")), F.lit(1.0)))
    .withColumn("label", F.col("default_next_month").cast("double"))
)
raw_cols = ["limit_bal", "sex", "education", "marriage", "age"] + [f"pay_{m}" for m in months] + [f"bill_amt{m}" for m in months] + [f"pay_amt{m}" for m in months]
derived = ["avg_bill", "avg_pay", "utilisation", "pay_ratio", "months_late", "max_months_late", "bill_trend"]
features = raw_cols + derived

train, test = feat.randomSplit([0.8, 0.2], seed=42)
train = train.cache(); test = test.cache()
print(f"train={train.count():,} test={test.count():,} positive rate={feat.agg(F.avg('label')).first()[0]:.3f}")

# COMMAND ----------

assembler = VectorAssembler(inputCols=features, outputCol="raw_features", handleInvalid="keep")
scaler = StandardScaler(inputCol="raw_features", outputCol="features")
auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
pr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")


def ks_statistic(pred):
    """Kolmogorov-Smirnov: max separation between cumulative distributions of scores for defaulters vs non-defaulters.
    The metric credit-risk teams actually quote."""
    from pyspark.ml.functions import vector_to_array

    s = pred.select(F.col("label"), vector_to_array("probability")[1].alias("p"))
    total = s.groupBy("label").count().collect()
    n1 = next(r["count"] for r in total if r["label"] == 1.0)
    n0 = next(r["count"] for r in total if r["label"] == 0.0)
    w = s.withColumn("d1", F.when(F.col("label") == 1.0, 1.0 / n1).otherwise(0.0)).withColumn("d0", F.when(F.col("label") == 0.0, 1.0 / n0).otherwise(0.0))
    from pyspark.sql import Window

    win = Window.orderBy(F.desc("p")).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    cum = w.withColumn("c1", F.sum("d1").over(win)).withColumn("c0", F.sum("d0").over(win))
    return cum.agg(F.max(F.abs(F.col("c1") - F.col("c0")))).first()[0]


def log_run(name, estimator, grid, extra_params=None):
    with mlflow.start_run(run_name=name) as run:
        cv = CrossValidator(estimator=Pipeline(stages=[assembler, scaler, estimator]), estimatorParamMaps=grid, evaluator=auc, numFolds=3, parallelism=2, seed=42)
        model = cv.fit(train)
        best = model.bestModel.stages[-1]
        best_params = model.getEstimatorParamMaps()[int(max(range(len(model.avgMetrics)), key=lambda i: model.avgMetrics[i]))]
        pred = model.transform(test)
        metrics = {"test_auc": auc.evaluate(pred), "test_pr_auc": pr.evaluate(pred), "test_ks": ks_statistic(pred), "cv_best_auc": max(model.avgMetrics)}
        mlflow.log_params({"algorithm": name, "n_features": len(features), "cv_folds": 3, **(extra_params or {})})
        mlflow.log_params({p.name: v for p, v in best_params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if hasattr(best, "featureImportances"):
            imp = sorted(zip(features, best.featureImportances.toArray(), strict=True), key=lambda x: -x[1])
            mlflow.log_dict({k: float(v) for k, v in imp}, "feature_importances.json")
            mlflow.log_text("\n".join(f"{k}: {v:.4f}" for k, v in imp[:15]), "top_features.txt")
        mlflow.spark.log_model(model.bestModel, "model")
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return run.info.run_id, metrics


lr_grid = ParamGridBuilder().addGrid(LogisticRegression.regParam, [0.01, 0.1]).addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5]).build()
lr_run, lr_metrics = log_run("logistic_regression", LogisticRegression(labelCol="label", maxIter=50), lr_grid)

gbt_grid = ParamGridBuilder().addGrid(GBTClassifier.maxDepth, [3, 5]).addGrid(GBTClassifier.maxIter, [40, 80]).build()
gbt_run, gbt_metrics = log_run("gradient_boosted_trees", GBTClassifier(labelCol="label", stepSize=0.1, seed=42), gbt_grid)

# COMMAND ----------

# MAGIC %md ## Register the winner
# MAGIC The registry is what turns "a notebook that worked once" into something a scoring job can reference by name and stage.

# COMMAND ----------

best_run = gbt_run if gbt_metrics["test_auc"] >= lr_metrics["test_auc"] else lr_run
mv = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "test_auc", f"{max(gbt_metrics['test_auc'], lr_metrics['test_auc']):.4f}")
print(f"registered {MODEL_NAME} v{mv.version} as @champion from run {best_run}")
