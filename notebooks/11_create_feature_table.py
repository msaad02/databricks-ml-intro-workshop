# Databricks notebook source
# MAGIC %md
# MAGIC # 11. Create a Unity Catalog Feature Engineering table
# MAGIC
# MAGIC This notebook promotes reusable PySpark customer features into a Feature Engineering table with a declared primary key. The next notebook uses a `FeatureLookup` to build its training set and `FeatureEngineeringClient.log_model` to retain feature metadata and lineage with the model.
# MAGIC
# MAGIC **Concepts:** Feature Engineering in Unity Catalog, primary keys, PySpark, pandas API on Spark, offline training lineage.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from databricks.feature_engineering import FeatureEngineeringClient

source_table = f"{catalog}.{schema}.customer_feature_source"
feature_table = f"{catalog}.{schema}.customer_features"

fe = FeatureEngineeringClient()
feature_df = spark.table(source_table)

display(feature_df.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create or refresh the feature table
# MAGIC
# MAGIC `customer_id` is the entity key. In a production time-series feature table, add timestamp keys and use point-in-time joins to prevent leakage.

# COMMAND ----------

if spark.catalog.tableExists(feature_table):
    fe.write_table(name=feature_table, df=feature_df, mode="merge")
    print(f"Merged refreshed features into {feature_table}")
else:
    fe.create_table(
        name=feature_table,
        primary_keys=["customer_id"],
        df=feature_df,
        description=(
            "Reusable customer features derived in PySpark from public samples.tpch "
            "customer and pre-cutoff order data."
        ),
    )
    print(f"Created Feature Engineering table {feature_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Spark, pandas API on Spark, and pandas
# MAGIC
# MAGIC - **PySpark DataFrame:** default for governed preparation and large data. Transformations run on Spark and write directly to Unity Catalog.
# MAGIC - **pandas API on Spark:** familiar pandas-style syntax while execution stays distributed on Spark.
# MAGIC - **pandas:** appropriate after aggregation or an explicit bounded sample when a model library expects in-memory data.
# MAGIC
# MAGIC Calling `toPandas()` on an unbounded feature table can exhaust the driver. The modeling notebook deliberately selects columns and caps the workshop sample first.

# COMMAND ----------

psdf = feature_df.pandas_api()
display(psdf.head(5).to_spark())

bounded_pdf = feature_df.orderBy("customer_id").limit(1000).toPandas()
display(bounded_pdf.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Where lineage appears
# MAGIC
# MAGIC 1. Catalog Explorer shows `samples.tpch` → `customer_feature_source` → `customer_features` table lineage.
# MAGIC 2. The propensity notebook creates a training set with `FeatureLookup`.
# MAGIC 3. `fe.log_model(..., training_set=training_set)` records the lookup specification with the MLflow model.
# MAGIC 4. Registering that model in Unity Catalog exposes the upstream feature-table relationship so scoring can reuse the same governed features.
