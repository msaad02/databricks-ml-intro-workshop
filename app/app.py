"""Step 6: a thin Databricks App over the model's output table.

Streamlit + a SQL warehouse read the predictions written in step 4.
Kept deliberately minimal — one query, one chart.
"""

import os

import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

# App resources are injected as env vars (see app.yaml / bundle).
CATALOG = os.getenv("CATALOG", "main")
SCHEMA = os.getenv("SCHEMA", "workshop")
TABLE = f"{CATALOG}.{SCHEMA}.predictions"

st.title("🚕 Taxi Fare Predictions")


@st.cache_data(ttl=600)
def load():
    cfg = Config()  # picks up the app's OAuth credentials automatically
    query = f"SELECT trip_distance, fare_amount, predicted_fare FROM {TABLE} LIMIT 1000"
    warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
    with sql.connect(
        server_hostname=cfg.host.removeprefix("https://"),
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        return conn.cursor().execute(query).fetchall_arrow().to_pandas()


df = load()
st.metric("Trips scored", len(df))
st.scatter_chart(df, x="trip_distance", y=["fare_amount", "predicted_fare"])
# Enrichment: add filters, wire an endpoint for on-demand scoring via Model
# Serving, or drop in Genie for natural-language Q&A over the table.
