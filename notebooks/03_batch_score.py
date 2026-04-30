# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Batch scoring from the registry
# MAGIC Loads whatever model is currently `@champion` — the scoring job never needs to know which run produced it —
# MAGIC scores the full customer table, writes probabilities to gold, and logs a score-distribution summary so drift is visible run to run.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

model = mlflow.spark.load_model(f"models:/{MODEL_NAME}@champion")
df = read(f"{LAKE}/bronze/credit_default")

months = range(1, 7)
feat = (
    df.withColumn("avg_bill", sum(F.col(f"bill_amt{m}") for m in months) / 6)
    .withColumn("avg_pay", sum(F.col(f"pay_amt{m}") for m in months) / 6)
    .withColumn("utilisation", F.col("avg_bill") / F.col("limit_bal"))
    .withColumn("pay_ratio", F.col("avg_pay") / F.greatest(F.col("avg_bill"), F.lit(1.0)))
    .withColumn("months_late", sum(F.when(F.col(f"pay_{m}") > 0, 1).otherwise(0) for m in months))
    .withColumn("max_months_late", F.greatest(*[F.col(f"pay_{m}") for m in months]))
    .withColumn("bill_trend", (F.col("bill_amt1") - F.col("bill_amt6")) / F.greatest(F.abs(F.col("bill_amt6")), F.lit(1.0)))
)
scored = model.transform(feat).withColumn("p_default", vector_to_array("probability")[1]).select("id", "limit_bal", "p_default", "default_next_month")
scored = scored.withColumn("risk_band", F.when(F.col("p_default") >= 0.5, "high").when(F.col("p_default") >= 0.25, "medium").otherwise("low"))
write(scored, f"{LAKE}/gold/credit_default_scores")

summary = scored.groupBy("risk_band").agg(F.count("*").alias("customers"), F.round(F.avg("p_default"), 3).alias("avg_p"), F.round(F.avg("default_next_month"), 3).alias("actual_default_rate")).orderBy("avg_p")
summary.show()
with mlflow.start_run(run_name="batch_score"):
    for r in summary.collect():
        mlflow.log_metric(f"band_{r['risk_band']}_customers", r["customers"])
        mlflow.log_metric(f"band_{r['risk_band']}_actual_rate", r["actual_default_rate"])
    mlflow.log_metric("scored_rows", scored.count())
