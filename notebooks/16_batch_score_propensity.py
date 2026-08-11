# Databricks notebook source
# MAGIC %md
# MAGIC # 16. Batch-score the registered propensity champion
# MAGIC
# MAGIC The registered model retains its Feature Engineering lineage. For portable serverless scoring, Spark joins the same governed feature table before a bounded pandas frame is passed to the `@champion` model.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

label_table = f"{catalog}.{schema}.customer_labels"
feature_table = f"{catalog}.{schema}.customer_features"
model_name = f"{catalog}.{schema}.propensity_to_convert"
output_table = f"{catalog}.{schema}.propensity_scores"

features = [
    "market_segment",
    "nation_key",
    "account_balance",
    "order_count",
    "historical_spend",
    "average_order_value",
    "days_since_last_order",
]

scoring = (
    spark.table(label_table)
    .select("customer_id")
    .join(spark.table(feature_table).select("customer_id", *features), "customer_id")
)
pdf = scoring.limit(50000).toPandas()

model_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_name}@champion"
)
model = mlflow.sklearn.load_model(f"{model_path}/data/feature_store/raw_model")
pdf["propensity_score"] = model.predict(pdf[features])

scored = spark.createDataFrame(pdf[["customer_id", "propensity_score"]]).withColumn(
    "scored_at", F.current_timestamp()
)

(
    scored.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(output_table)
)

display(scored.orderBy(F.col("propensity_score").desc()).limit(20))
print(f"Wrote {scored.count():,} scores to {output_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC Inspect the registered model version and `propensity_scores` table in Catalog Explorer. The model's Feature Engineering metadata and Unity Catalog lineage explain which governed features produced the scores.
