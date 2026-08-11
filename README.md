# Databricks ML Intro Workshop — cheat sheet

An end-to-end Databricks data-science workshop built as a [Databricks Asset Bundle](https://docs.databricks.com/dev-tools/bundles/index.html). The repository is a reference: participants should build their own small version, ideally with AI help, rather than copy every line.

There are two paths:

- **Core:** a short NYC taxi regression that teaches the platform lifecycle.
- **Marketing science:** optional notebooks that add Feature Engineering and examples of propensity, forecasting, MMM, and incrementality without turning the main workshop into a production codebase.

See [DEMO.md](DEMO.md) for the architecture and experiment-design rationale.

## Core path

| Step | File | Concepts |
|---|---|---|
| Find data | `notebooks/01_explore_catalog.ipynb` | Unity Catalog, samples |
| Explore | `notebooks/02_eda.ipynb` | notebooks, SQL, Spark, pandas |
| Train and register | `notebooks/03_train_register.ipynb` | MLflow, UC models |
| Batch predict | `notebooks/04_batch_predict.ipynb` | jobs, scheduling, Delta |
| Productionalize | `databricks.yml`, `resources/`, `.github/` | DAB, CI/CD |
| Thin app | `app/` | Databricks Apps |

## Marketing-science path

This path uses public `samples.tpch` customer and order data. PySpark creates the reusable governed tables. Marketing spend and treatment fields are deterministic simulations because TPC-H does not contain campaign data.

| Step | File | Main idea |
|---|---|---|
| Prepare data | `notebooks/10_prepare_marketing_data.py` | PySpark, temporal labels, Delta lineage |
| Create features | `notebooks/11_create_feature_table.py` | UC Feature Engineering table, pandas API on Spark |
| Propensity | `notebooks/12_propensity_models.py` | Logistic regression vs. random forest; feature-aware champion |
| Forecasting | `notebooks/13_forecasting_models.py` | Ridge vs. random forest; chronological validation |
| MMM | `notebooks/14_mmm_models.py` | Ridge vs. Bayesian ridge; channel coefficients |
| Incrementality | `notebooks/15_incrementality_models.py` | Difference in means vs. T-learner uplift |
| Batch score | `notebooks/16_batch_score_propensity.py` | `@champion` alias and governed feature reuse |

### How experiments are organized

Use one experiment per business question. Put comparable algorithms inside it as runs:

```text
propensity:      logistic_regression, random_forest
forecasting:     ridge, random_forest
marketing-mix:   ridge, bayesian_ridge
incrementality:  difference_in_means, t_learner_random_forest
```

Propensity and forecasting should not share an experiment because their targets, validation strategies, and metrics are different.

### Spark and pandas

- Use **PySpark** for source joins, feature preparation, and Unity Catalog writes.
- Use **pandas API on Spark** when pandas syntax is useful but execution should stay distributed.
- Use **pandas** only after aggregation or an explicit bounded sample for an in-memory model library.

The propensity example uses `FeatureEngineeringClient.create_training_set` and `log_model`, which retains the feature lookups and upstream lineage with the registered model.

## Two jobs

- `batch-inference` is the small Jobs lesson: train the fare model, run scheduled batch prediction, and write the app's table.
- `all-workshop-examples` runs notebooks `01–04` and `10–16` in order, so the complete workshop needs one click instead of eleven **Run All** actions.

## Run it

```bash
# Authenticate to the target workspace.
databricks auth login --host <workspace-url> --profile <your-profile>

# Deploy the schema, notebooks, jobs, and app.
databricks bundle deploy -t dev -p <your-profile> \
  --var="catalog=<your_catalog>" --var="schema=<your_schema>"

# Run the small Jobs demo or every notebook, then deploy the thin app.
databricks bundle run batch_inference -t dev -p <your-profile>
databricks bundle run all_examples -t dev -p <your-profile>
databricks bundle run fare_app -t dev -p <your-profile>
```

`databricks bundle validate -t dev` checks the bundle before deployment.
The app looks up a warehouse named `Serverless Starter Warehouse`; override
`warehouse_id` at deployment time if your workspace uses a different name.

## What gets created

Resources are tagged `workshop=databricks-ml-intro`. Development-mode resource names are also prefixed with the current username.

| Asset | Name |
|---|---|
| Bundle | `ml-workshop` |
| Schema | `<catalog>.<schema>` |
| Scheduled Jobs demo | `batch-inference` |
| All-notebooks job | `all-workshop-examples` |
| App | `fare-app` |
| Feature table | `<catalog>.<schema>.customer_features` |
| Propensity model | `<catalog>.<schema>.propensity_to_convert` |
| Propensity scores | `<catalog>.<schema>.propensity_scores` |

## Deliberate limits

The notebooks demonstrate the shape of each workflow without implementing full production MMM, causal validation, multi-horizon forecasting, online Feature Serving, or real-time Model Serving. Those are follow-on topics, not prerequisites for understanding MLflow.
