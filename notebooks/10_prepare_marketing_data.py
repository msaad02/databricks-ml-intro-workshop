# Databricks notebook source
# MAGIC %md
# MAGIC # 10. Prepare the marketing-science tables
# MAGIC
# MAGIC We use public `samples.tpch` customer and order data. PySpark keeps preparation distributed and writes every reusable result to Unity Catalog.
# MAGIC
# MAGIC TPC-H has no campaign spend or treatment assignment, so the MMM and incrementality fields are clearly labeled simulations. They make the modeling patterns observable without pretending to be customer data.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from pyspark.sql import functions as F
from pyspark.sql.window import Window


def save(df, table):
    name = f"{catalog}.{schema}.{table}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
    print(f"Wrote {name}")


customers = spark.table("samples.tpch.customer").select(
    F.col("c_custkey").cast("long").alias("customer_id"),
    F.col("c_mktsegment").alias("market_segment"),
    F.col("c_nationkey").cast("int").alias("nation_key"),
    F.col("c_acctbal").cast("double").alias("account_balance"),
)

orders = spark.table("samples.tpch.orders").select(
    F.col("o_orderkey").cast("long").alias("order_id"),
    F.col("o_custkey").cast("long").alias("customer_id"),
    F.to_date("o_orderdate").alias("order_date"),
    F.col("o_totalprice").cast("double").alias("order_value"),
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Customer features and a future label
# MAGIC
# MAGIC Features stop at 1995-01-01. The label asks whether the customer orders during 1995, preventing future information from leaking into training.

# COMMAND ----------

history = orders.filter("order_date < '1995-01-01'")
future = orders.filter("order_date >= '1995-01-01' AND order_date < '1996-01-01'")

aggregates = history.groupBy("customer_id").agg(
    F.count("order_id").cast("double").alias("order_count"),
    F.sum("order_value").alias("historical_spend"),
    F.avg("order_value").alias("average_order_value"),
    F.max("order_date").alias("last_order_date"),
)

feature_source = (
    customers.join(aggregates, "customer_id", "left")
    .fillna({"order_count": 0.0, "historical_spend": 0.0, "average_order_value": 0.0})
    .withColumn(
        "days_since_last_order",
        F.coalesce(
            F.datediff(F.lit("1995-01-01"), "last_order_date"), F.lit(9999)
        ).cast("double"),
    )
    .drop("last_order_date")
)

labels = (
    customers.select("customer_id")
    .join(
        future.select("customer_id")
        .distinct()
        .withColumn("converted_next_year", F.lit(1)),
        "customer_id",
        "left",
    )
    .fillna({"converted_next_year": 0})
)

save(feature_source, "customer_feature_source")
save(labels, "customer_labels")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Weekly series for forecasting

# COMMAND ----------

weekly = (
    orders.groupBy(F.date_trunc("week", "order_date").cast("date").alias("week_start"))
    .agg(
        F.sum("order_value").alias("weekly_revenue"),
        F.count("order_id").cast("double").alias("weekly_orders"),
    )
    .orderBy("week_start")
)
save(weekly, "weekly_commerce")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Simulated weekly media for MMM
# MAGIC
# MAGIC Each channel has simple carryover (adstock) and diminishing returns (saturation). The outcome is generated from known channel effects, seasonality, and noise so the regression has something meaningful to recover.

# COMMAND ----------

w = Window.orderBy("week_start")
mmm = (
    weekly.withColumn("week_index", F.row_number().over(w).cast("double"))
    .withColumn(
        "search_spend",
        60000 + 15000 * F.sin(F.col("week_index") / 5) + 5000 * F.rand(1),
    )
    .withColumn(
        "social_spend",
        35000 + 10000 * F.cos(F.col("week_index") / 7) + 4000 * F.rand(2),
    )
    .withColumn(
        "video_spend",
        50000 + 18000 * F.sin(F.col("week_index") / 11) + 6000 * F.rand(3),
    )
)

for channel, decay, scale in [
    ("search", 0.55, 110000.0),
    ("social", 0.35, 65000.0),
    ("video", 0.70, 150000.0),
]:
    adstock = F.col(f"{channel}_spend") + decay * F.coalesce(
        F.lag(f"{channel}_spend").over(w), F.lit(0.0)
    )
    mmm = mmm.withColumn(f"{channel}_transformed", 1 - F.exp(-adstock / scale))

mmm = (
    mmm.withColumn(
        "promotion_week", (F.pmod("week_index", F.lit(13.0)) == 0).cast("double")
    )
    .withColumn(
        "season", F.sin(2 * F.lit(3.141592653589793) * F.col("week_index") / 52)
    )
    .withColumn(
        "simulated_signups",
        18000
        + 7500 * F.col("search_transformed")
        + 4200 * F.col("social_transformed")
        + 6100 * F.col("video_transformed")
        + 2500 * F.col("promotion_week")
        + 1600 * F.col("season")
        + 600 * (F.rand(4) - 0.5),
    )
    .select(
        "week_start",
        "search_transformed",
        "social_transformed",
        "video_transformed",
        "promotion_week",
        "season",
        "simulated_signups",
    )
)
save(mmm, "mmm_weekly")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Simulated randomized campaign for incrementality

# COMMAND ----------

campaign = (
    feature_source.withColumn(
        "treatment", (F.pmod(F.xxhash64("customer_id"), F.lit(2)) == 0).cast("int")
    )
    .withColumn(
        "true_uplift",
        0.025
        + F.when(F.col("order_count") >= 8, 0.035).otherwise(0.0)
        + F.when(F.col("market_segment") == "HOUSEHOLD", 0.02).otherwise(0.0),
    )
    .withColumn(
        "probability",
        F.least(
            F.lit(0.08)
            + F.least(F.col("order_count") / 100, F.lit(0.12))
            + F.col("treatment") * F.col("true_uplift"),
            F.lit(0.95),
        ),
    )
    .withColumn("converted", (F.rand(42) < F.col("probability")).cast("int"))
    .drop("probability")
)
save(campaign, "campaign_experiment")

# COMMAND ----------
# MAGIC %md
# MAGIC Open an output table in Catalog Explorer and select **Lineage** to see the PySpark read/write path from `samples.tpch`. The next notebooks add table-to-run and table-to-model lineage with MLflow and Feature Engineering.
