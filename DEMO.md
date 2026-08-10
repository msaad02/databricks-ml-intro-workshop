# Databricks ML intro workshop

## Purpose

This workshop gives a Python-strong data-science team a hands-on path from governed data to reproducible experiments, registered models, scheduled batch scoring, and stakeholder-facing outputs. The examples use public `samples.tpch` data plus clearly labeled, deterministic marketing fields created in PySpark. No customer data or customer branding is embedded in the repository.

## Learning outcomes

Participants will:

1. Discover governed source data in Unity Catalog.
2. Build reusable Delta and Feature Engineering tables with PySpark.
3. See the distinction between Spark, pandas API on Spark, and bounded pandas conversion.
4. Organize MLflow experiments around business questions rather than algorithms.
5. Compare multiple candidate models within an experiment.
6. Record parameters, metrics, diagnostic plots, source-table lineage, and model artifacts.
7. Register the selected model in Unity Catalog and use an alias for batch scoring.
8. Understand how propensity, forecasting, marketing mix modeling, and incrementality differ.

## Experiment design

An MLflow experiment should contain runs that answer the same business question with the same target and comparable evaluation metrics. Therefore, the workshop uses four experiments:

| Experiment | Question | Runs within the experiment |
|---|---|---|
| Propensity | Which customers are likely to convert? | Logistic regression and random forest |
| Forecasting | What will weekly commerce activity look like next? | Ridge regression and random forest |
| Marketing mix modeling | How much outcome is associated with each marketing channel? | Ridge and Bayesian ridge |
| Incrementality | What happened because of treatment? | Randomized difference-in-means and a two-model uplift learner |

Different algorithms for one target belong in the same experiment because their metrics are comparable. Propensity and forecasting do not belong in the same experiment because their targets, validation designs, metrics, and artifacts are different.

## Data design

The public `samples.tpch.customer` and `samples.tpch.orders` tables provide customer, segment, order-date, and revenue signals that NYC taxi lacks.

The preparation notebook creates the following governed tables:

| Table | Purpose | Provenance |
|---|---|---|
| `customer_labels` | Temporal conversion label and scoring keys | TPC-H customer and orders |
| `weekly_commerce` | Weekly revenue and order-count series | TPC-H orders |
| `mmm_weekly` | Marketing-spend, transformed-channel, and outcome series | Weekly TPC-H baseline plus deterministic simulated marketing fields |
| `campaign_experiment` | Randomized treatment/control example with known heterogeneous lift | Customer features plus deterministic simulated treatment/outcome |
| `customer_features` | Reusable customer feature table with a primary key | TPC-H customer and pre-cutoff orders |

The simulated fields exist because public sample data does not contain real media-spend or controlled-treatment assignments. Every notebook labels this boundary explicitly.

## Architecture

```text
samples.tpch.customer + samples.tpch.orders
                     |
                     v
       PySpark preparation and aggregation
                     |
          +----------+----------+
          |                     |
          v                     v
Unity Catalog Delta tables   UC Feature Engineering table
          |                     |
          +----------+----------+
                     v
       Four question-specific MLflow experiments
                     |
          candidate runs + diagnostics
                     |
                     v
       UC registered propensity champion
                     |
                     v
            scheduled batch scoring
```

## Lineage

The workshop demonstrates lineage at two complementary levels:

- `mlflow.log_input(mlflow.data.load_delta(...))` attaches governed Delta inputs to non-feature models and their runs.
- `FeatureEngineeringClient.create_training_set(...)` plus `FeatureEngineeringClient.log_model(...)` records the feature lookups used by the propensity model. The registered Unity Catalog model retains feature metadata and upstream-table lineage.

## Delivery principles

- Use serverless compute and Unity Catalog only.
- Keep data preparation distributed in PySpark.
- Convert to pandas only after aggregation or bounded sampling for libraries that expect in-memory data.
- Show pandas API on Spark as a familiar syntax option without moving the full dataset to the driver.
- Prefer transparent, workshop-sized models over dependency-heavy production frameworks.
- Demonstrate a lifecycle concept once rather than repeating registry and deployment ceremony in every notebook.
- Treat the Bayesian-ridge MMM as a conceptual bridge to full probabilistic MMM frameworks such as PyMC-Marketing, not as a claim that the example is production MMM.

## Out of scope

- Production causal validation or media-budget optimization
- Real customer media and conversion data
- Online Feature Serving or real-time Model Serving
- Full Bayesian MCMC at marketing-production scale
- Customer-specific governance and privacy controls
