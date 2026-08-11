# Databricks notebook source
# MAGIC %md
# MAGIC # 14. Marketing mix modeling: interpretable aggregate regression
# MAGIC
# MAGIC MMM rows represent weeks, not customers. The upstream PySpark notebook already applied simple adstock and saturation transforms. Here we compare a regularized regression with a lightweight Bayesian regression.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/databricks-ml-intro-workshop/marketing-mix")
source = f"{catalog}.{schema}.mmm_weekly"

pdf = spark.table(source).orderBy("week_start").toPandas()
features = [
    "search_transformed",
    "social_transformed",
    "video_transformed",
    "promotion_week",
    "season",
]
channels = features[:3]
split = int(len(pdf) * 0.80)
train, test = pdf.iloc[:split], pdf.iloc[split:]
X_train, y_train = train[features], train["simulated_signups"]
X_test, y_test = test[features], test["simulated_signups"]

candidates = {"ridge": Ridge(alpha=1.0), "bayesian_ridge": BayesianRidge()}
results = []

for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, prediction),
            "rmse": mean_squared_error(y_test, prediction) ** 0.5,
            "r2": r2_score(y_test, prediction),
        }
        coefficients = {
            feature: float(coefficient)
            for feature, coefficient in zip(features, model.coef_)
        }
        contributions = {
            channel: float((pdf[channel] * coefficients[channel]).mean())
            for channel in channels
        }
        mlflow.log_param("candidate", name)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(coefficients, "coefficients.json")
        mlflow.log_dict(contributions, "average_channel_contributions.json")
        mlflow.log_input(mlflow.data.load_delta(table_name=source), context="training")
        mlflow.sklearn.log_model(
            model, artifact_path="model", input_example=X_train.head(5)
        )
        results.append({"candidate": name, "run_id": run.info.run_id, **metrics})

display(pd.DataFrame(results).sort_values("rmse"))

# COMMAND ----------
# MAGIC %md
# MAGIC The Bayesian-ridge run is an approachable bridge, not a production Bayesian MMM. A full implementation would add explicit priors, posterior sampling, credible intervals, posterior-predictive checks, and response curves—without changing the basic MLflow pattern of logging assumptions, diagnostics, and a reproducible model artifact.
