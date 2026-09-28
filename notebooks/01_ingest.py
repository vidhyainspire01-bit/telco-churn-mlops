# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 - Ingest: raw CSV -> bronze -> silver
# MAGIC - **Bronze**: file as-is (all strings) + ingestion metadata. Never cleaned, so we can always replay.
# MAGIC - **Silver**: typed, cleaned, one row per customer, `label` = 1 if churned.
# MAGIC - Fails the run on data-quality violations (this is what the CI gate relies on later).

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("schema", "telco")
dbutils.widgets.text("raw_path", "/Volumes/dev_ml/telco/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_path = dbutils.widgets.get("raw_path")

bronze_table = f"{catalog}.{schema}.bronze_customers"
silver_table = f"{catalog}.{schema}.silver_customers"
print(f"raw={raw_path}\nbronze={bronze_table}\nsilver={silver_table}")

# COMMAND ----------

# MAGIC %md ## Bronze

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "false")          # keep raw strings in bronze
    .load(raw_path)
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp())
)

# Kaggle file is a full snapshot, so overwrite (not append) each run
(bronze_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table))
print(f"bronze rows: {spark.table(bronze_table).count()}")

# COMMAND ----------

# MAGIC %md ## Silver

# COMMAND ----------

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

b = spark.table(bronze_table)

silver_df = (
    b.select(
        F.trim("customerID").alias("customerID"),
        *[F.trim(F.col(c)).alias(c) for c in categorical_cols],   # keep raw strings; model pipeline encodes
        F.col("SeniorCitizen").cast("int").alias("SeniorCitizen"),
        F.col("tenure").cast("int").alias("tenure"),
        F.col("MonthlyCharges").cast("double").alias("MonthlyCharges"),
        # 11 rows have blank TotalCharges; all have tenure = 0 (brand-new customers) -> 0.0 is correct
        F.when(F.trim("TotalCharges") == "", F.lit(0.0))
         .otherwise(F.col("TotalCharges").cast("double")).alias("TotalCharges"),
        F.when(F.trim("Churn") == "Yes", 1)
         .when(F.trim("Churn") == "No", 0).alias("label"),
        "_ingested_at",
    )
    .dropDuplicates(["customerID"])
)

# COMMAND ----------

# MAGIC %md ## Data-quality checks (fail the run if violated)

# COMMAND ----------

total = silver_df.count()
checks = {
    "non_empty": total > 0,
    "no_null_label": silver_df.filter(F.col("label").isNull()).count() == 0,
    "unique_customer_id": silver_df.select("customerID").distinct().count() == total,
    "no_null_numeric": silver_df.filter(
        F.col("tenure").isNull() | F.col("MonthlyCharges").isNull() | F.col("TotalCharges").isNull()
    ).count() == 0,
    "tenure_non_negative": silver_df.filter(F.col("tenure") < 0).count() == 0,
}

for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")

failed = [n for n, ok in checks.items() if not ok]
if failed:
    raise ValueError(f"Data-quality checks failed: {failed}")

# COMMAND ----------

(silver_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table))

churn_rate = spark.table(silver_table).agg(F.avg("label")).first()[0]
print(f"silver rows: {total}, churn rate: {churn_rate:.3f}")