# Databricks notebook source
# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.
# MAGIC %md
# MAGIC # 16. Batch-score propensity with automatic feature lookup
# MAGIC
# MAGIC The registered propensity champion retained its Feature Engineering training-set specification. Batch scoring therefore supplies customer keys; Feature Engineering retrieves the same governed feature columns and passes them to the model.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

label_table = f"{catalog}.{schema}.customer_labels"
model_name = f"{catalog}.{schema}.propensity_to_convert"
output_table = f"{catalog}.{schema}.propensity_scores"

fe = FeatureEngineeringClient()

# The scoring frame intentionally contains only the lookup key.
scoring_keys = spark.table(label_table).select("customer_id")
scored = fe.score_batch(
    model_uri=f"models:/{model_name}@champion",
    df=scoring_keys,
)

prediction_column = "prediction"
if prediction_column in scored.columns:
    scored = scored.withColumnRenamed(prediction_column, "propensity_score")

scored = scored.withColumn("scored_at", F.current_timestamp())

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
# MAGIC Inspect the registered model version and `propensity_scores` table in Catalog Explorer. The model's feature lookup and Unity Catalog lineage explain which governed features produced the scores.
