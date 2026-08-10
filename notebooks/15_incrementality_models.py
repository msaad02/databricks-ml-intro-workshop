# Databricks notebook source
# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.
# MAGIC %md
# MAGIC # 15. Incrementality: population lift and customer-level uplift
# MAGIC
# MAGIC A propensity model predicts who converts. Incrementality asks what happened **because of treatment**. This experiment contains one statistical estimate and one T-learner uplift model.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/databricks-ml-intro-workshop/incrementality")
source = f"{catalog}.{schema}.campaign_experiment"

features = [
    "nation_key",
    "account_balance",
    "order_count",
    "historical_spend",
    "average_order_value",
    "days_since_last_order",
]
pdf = (
    spark.table(source)
    .select(*features, "treatment", "converted", "true_uplift")
    .limit(50000)
    .toPandas()
)
train, test = train_test_split(pdf, test_size=0.30, random_state=42)


def lift(frame):
    treated = frame.loc[frame.treatment == 1, "converted"]
    control = frame.loc[frame.treatment == 0, "converted"]
    effect = treated.mean() - control.mean()
    standard_error = (
        treated.var() / len(treated) + control.var() / len(control)
    ) ** 0.5
    return (
        float(effect),
        float(effect - 1.96 * standard_error),
        float(effect + 1.96 * standard_error),
    )


effect, low, high = lift(test)
with mlflow.start_run(run_name="randomized_difference_in_means"):
    mlflow.log_metrics(
        {"incremental_lift": effect, "lift_ci_low": low, "lift_ci_high": high}
    )
    mlflow.log_param("estimator", "difference_in_means")
    mlflow.log_input(mlflow.data.load_delta(table_name=source), context="experiment")

# COMMAND ----------
# MAGIC %md
# MAGIC ## T-learner
# MAGIC
# MAGIC One model learns outcomes for treated customers and another for controls. Their probability difference is predicted uplift.

# COMMAND ----------

control = RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1
)
treated = RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_leaf=30, random_state=43, n_jobs=-1
)

control_train = train[train.treatment == 0]
treated_train = train[train.treatment == 1]
control.fit(control_train[features], control_train.converted)
treated.fit(treated_train[features], treated_train.converted)

test = test.copy()
test["predicted_uplift"] = (
    treated.predict_proba(test[features])[:, 1]
    - control.predict_proba(test[features])[:, 1]
)
top_20 = test.nlargest(max(100, int(len(test) * 0.20)), "predicted_uplift")
top_effect, _, _ = lift(top_20)

with mlflow.start_run(run_name="t_learner_random_forest"):
    mlflow.log_metrics(
        {
            "population_incremental_lift": effect,
            "uplift_at_top_20pct": top_effect,
            "predicted_true_uplift_correlation": test.predicted_uplift.corr(
                test.true_uplift
            ),
        }
    )
    mlflow.log_param("estimator", "two_random_forests")
    mlflow.log_input(mlflow.data.load_delta(table_name=source), context="training")
    mlflow.sklearn.log_model(
        control, artifact_path="control_model", input_example=train[features].head(5)
    )
    mlflow.sklearn.log_model(
        treated, artifact_path="treatment_model", input_example=train[features].head(5)
    )

display(
    pd.DataFrame(
        [
            {"estimator": "difference_in_means", "lift": effect},
            {"estimator": "T-learner top 20%", "lift": top_effect},
        ]
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC The first run correctly has no model artifact: not every valid data-science result is a deployable model. The second run stores two response models because uplift is the difference between two potential outcomes.
