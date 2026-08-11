# Databricks notebook source
# MAGIC %md
# MAGIC # 12. Propensity: compare two models in one MLflow experiment
# MAGIC
# MAGIC Both runs answer the same question—who will convert?—using the same label and metrics. That is why logistic regression and random forest belong in one experiment.

# COMMAND ----------

# ruff: noqa: F821  # Databricks injects spark, dbutils, and display.

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "workshop")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")

import mlflow
import mlflow.sklearn
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/databricks-ml-intro-workshop/propensity")

feature_table = f"{catalog}.{schema}.customer_features"
label_table = f"{catalog}.{schema}.customer_labels"
model_name = f"{catalog}.{schema}.propensity_to_convert"

features = [
    "market_segment",
    "nation_key",
    "account_balance",
    "order_count",
    "historical_spend",
    "average_order_value",
    "days_since_last_order",
]

fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
    df=spark.table(label_table),
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table, lookup_key="customer_id", feature_names=features
        )
    ],
    label="converted_next_year",
    exclude_columns=["customer_id"],
)

# Preparation remains in Spark; only this explicit workshop-sized sample moves to pandas.
pdf = training_set.load_df().limit(50000).toPandas()
X, y = pdf[features], pdf["converted_next_year"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# COMMAND ----------

numeric = [column for column in features if column != "market_segment"]
prepare = ColumnTransformer(
    [
        ("numeric", StandardScaler(), numeric),
        ("segment", OneHotEncoder(handle_unknown="ignore"), ["market_segment"]),
    ]
)

candidates = {
    "logistic_regression": make_pipeline(
        clone(prepare),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    ),
    "random_forest": make_pipeline(
        clone(prepare),
        RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

results, fitted = [], {}
for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        metrics = {
            "average_precision": average_precision_score(y_test, probability),
            "roc_auc": roc_auc_score(y_test, probability),
            "log_loss": log_loss(y_test, probability),
        }
        mlflow.log_param("candidate", name)
        mlflow.log_metrics(metrics)
        mlflow.log_input(
            mlflow.data.load_delta(table_name=feature_table), context="features"
        )
        mlflow.log_input(
            mlflow.data.load_delta(table_name=label_table), context="labels"
        )
        mlflow.sklearn.log_model(
            model, artifact_path="model", input_example=X_train.head(5)
        )
        results.append({"candidate": name, "run_id": run.info.run_id, **metrics})
        fitted[name] = model

results = pd.DataFrame(results).sort_values("average_precision", ascending=False)
display(results)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register the champion with its feature lookups
# MAGIC
# MAGIC The small wrapper makes `predict()` return a probability. `fe.log_model` records the training-set lookup and upstream feature lineage with the registered model.

# COMMAND ----------


class ProbabilityModel(BaseEstimator):
    def __init__(self, classifier):
        self.classifier = classifier

    def predict(self, frame):
        return self.classifier.predict_proba(frame)[:, 1]


best_name = results.iloc[0]["candidate"]
with mlflow.start_run(run_name=f"champion_{best_name}"):
    info = fe.log_model(
        model=ProbabilityModel(fitted[best_name]),
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=model_name,
        input_example=X_train.head(5),
    )

client = MlflowClient()
version = getattr(info, "registered_model_version", None)
if version is None:
    versions = client.search_model_versions(f"name = '{model_name}'")
    version = max(versions, key=lambda item: int(item.version)).version
client.set_registered_model_alias(model_name, "champion", version)
print(f"Registered {model_name} version {version} as @champion")
