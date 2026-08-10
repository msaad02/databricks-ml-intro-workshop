# Workshop build tasks

## Design

- [x] Inspect the existing notebook, bundle, job, and app structure.
- [x] Choose `samples.tpch` as the governed source and document simulated marketing fields.
- [x] Define one MLflow experiment per business question and candidate runs per algorithm.

## Implementation

- [x] Add PySpark preparation for customer labels, weekly commerce, MMM, and treatment/control tables.
- [x] Add a Unity Catalog Feature Engineering table and training-set lineage example.
- [x] Add propensity candidate models and register the champion with feature lineage.
- [x] Add forecasting candidate models with time-aware validation.
- [x] Add marketing-mix candidate models and channel-contribution artifacts.
- [x] Add randomized incrementality and uplift-model examples.
- [x] Add scheduled batch scoring through the registered propensity champion.
- [x] Update bundle resources, dependencies, and workshop documentation.

## Validation

- [x] Parse every notebook and compile Python cells where applicable.
- [x] Validate YAML and parse the Databricks Asset Bundle locally.
- [x] Run pure-Python model checks locally with `uv`.
- [x] Deploy the bundle, execute the core job, and verify the app in a workspace.
