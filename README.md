# Databricks ML Intro Workshop — cheat sheet

An end-to-end Databricks data science flow, built as a [Databricks Asset Bundle (DAB)](https://docs.databricks.com/dev-tools/bundles/index.html). This is the **reference** — the workshop goal is to build your own version (ideally with AI help), using this as inspiration.

Each workshop step is its own file. Nothing here is complex on purpose; look for the `> Expand here:` notes for how each step extends to real workloads.

## The flow

| Step | File | Databricks concepts |
|---|---|---|
| 1. Find data | `notebooks/01_explore_catalog.ipynb` | Unity Catalog, `samples` |
| 2. EDA | `notebooks/02_eda.ipynb` | notebooks, SQL, compute |
| 3. Train + register | `notebooks/03_train_register.ipynb` | MLflow, UC registered models |
| 4. Batch predict | `notebooks/04_batch_predict.ipynb` | jobs, scheduling, Delta |
| 5. Productionalize | `databricks.yml`, `resources/`, `.github/` | CI/CD, DAB |
| 6. Thin app | `app/` | Databricks Apps |

## Run it (step 5: this whole repo *is* the productionalization story)

```bash
# 0. Set the `host` in databricks.yml to your own workspace URL

# 1. Authenticate to the workspace
databricks auth login --profile <your-profile>

# 2. Deploy the bundle (schema + job + app) to your dev target
databricks bundle deploy -t dev -p <your-profile> \
  --var="catalog=<your_catalog>" --var="schema=<your_schema>"

# 3. Run notebooks 1–3 interactively in the workspace (they were deployed above)

# 4. Run the batch job
databricks bundle run batch_inference -t dev -p <your-profile>

# 5. Deploy the app (needs a SQL warehouse id)
databricks bundle deploy -t dev -p <your-profile> --var="warehouse_id=<id>"
```

`databricks bundle validate -t dev` checks everything before you deploy.

## What gets created

Every resource is tagged `workshop=databricks-ml-intro`, so you can find and clean up
workshop assets in a shared workspace. In the `dev` target, names are also prefixed
with your username (e.g. `[dev jane_doe] batch-inference`).

| Asset | Name |
|---|---|
| Bundle | `ml-workshop` |
| Schema | `<catalog>.workshop` |
| Job | `batch-inference` |
| App | `fare-app` |
| Registered model | `<catalog>.workshop.fare_model` |
| Output table | `<catalog>.workshop.predictions` |

## Out of scope (talk about, don't build)

Model Serving, Feature Store, in-depth governance, advanced ETL, LDP.
