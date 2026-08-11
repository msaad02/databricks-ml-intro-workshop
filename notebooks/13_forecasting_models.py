# Databricks notebook source
# MAGIC %md
# MAGIC # 13. Forecasting: compare models with a chronological holdout
# MAGIC
# MAGIC Forecasting gets its own experiment because its target and metrics are different from propensity. The last 20% of weeks form the test set; random splitting would leak the future.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/databricks-ml-intro-workshop/forecasting")
source = f"{catalog}.{schema}.weekly_commerce"

pdf = spark.table(source).orderBy("week_start").toPandas()
pdf["week_start"] = pd.to_datetime(pdf["week_start"])
pdf["lag_1"] = pdf["weekly_revenue"].shift(1)
pdf["lag_4"] = pdf["weekly_revenue"].shift(4)
pdf["rolling_4"] = pdf["weekly_revenue"].shift(1).rolling(4).mean()
pdf["orders_lag_1"] = pdf["weekly_orders"].shift(1)
pdf["week"] = pdf["week_start"].dt.isocalendar().week.astype(float)
pdf["season_sin"] = np.sin(2 * np.pi * pdf["week"] / 52)
pdf["season_cos"] = np.cos(2 * np.pi * pdf["week"] / 52)
pdf = pdf.dropna()

features = [
    "lag_1",
    "lag_4",
    "rolling_4",
    "orders_lag_1",
    "season_sin",
    "season_cos",
]
split = int(len(pdf) * 0.80)
train, test = pdf.iloc[:split], pdf.iloc[split:]
X_train, y_train = train[features], train["weekly_revenue"]
X_test, y_test = test[features], test["weekly_revenue"]

candidates = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1
    ),
}

results = []
for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, prediction),
            "rmse": mean_squared_error(y_test, prediction) ** 0.5,
            "wmape": np.abs(y_test - prediction).sum() / np.abs(y_test).sum(),
        }
        mlflow.log_param("candidate", name)
        mlflow.log_metrics(metrics)
        mlflow.log_input(mlflow.data.load_delta(table_name=source), context="training")
        mlflow.sklearn.log_model(
            model, artifact_path="model", input_example=X_train.head(5)
        )
        results.append({"candidate": name, "run_id": run.info.run_id, **metrics})

display(pd.DataFrame(results).sort_values("rmse"))

# COMMAND ----------
# MAGIC %md
# MAGIC A production version would add rolling backtests and multi-horizon forecasts. Those are useful next steps, but unnecessary for demonstrating how model alternatives and time-aware metrics appear in MLflow.
