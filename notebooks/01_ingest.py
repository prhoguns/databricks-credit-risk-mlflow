# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest: UCI "Default of Credit Card Clients"
# MAGIC 30,000 Taiwanese credit-card customers, April–September 2005: credit limit, demographics, six months of
# MAGIC repayment status, bill and payment amounts, and whether they defaulted the following month (22.1% did).
# MAGIC Source: https://archive.ics.uci.edu/dataset/350 (CC BY 4.0). Landed as-is into bronze.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import io
import zipfile

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
local_zip = os.getenv("CREDIT_ZIP")  # tests pass a pre-downloaded copy

raw_bytes = open(local_zip, "rb").read() if local_zip else requests.get(URL, timeout=120).content
with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
    xls = z.read(next(n for n in z.namelist() if n.endswith(".xls")))
pdf = pd.read_excel(io.BytesIO(xls), header=1)
pdf.columns = [c.strip().lower().replace(" ", "_") for c in pdf.columns]
pdf = pdf.rename(columns={"default_payment_next_month": "default_next_month", "pay_0": "pay_1"})
print(pdf.shape, "default rate:", round(pdf["default_next_month"].mean(), 4))

# COMMAND ----------

bronze = spark.createDataFrame(pdf)  # noqa: F821
write(bronze, f"{LAKE}/bronze/credit_default")
print("bronze rows:", bronze.count())
