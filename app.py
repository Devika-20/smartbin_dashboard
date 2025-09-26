import json, time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from bin_data_generator import ensure_bins_master, simulate_history, ensure_weather_profiles, DATA_DIR, HISTORY_PATH
from model_training import train_models, load_data
from inference import predict_overflow_and_fill
from notifier import notify_municipality
from utils import append_jsonl, now_iso, human_pct

st.set_page_config(page_title="SmartBin – Overflow Prediction", layout="wide")

st.title("SmartBin – Overflow Prediction Dashboard")
st.caption("Software-only prototype with simulated data, ML models, and live alerts.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    retrain = st.checkbox("Retrain models on startup", value=False)
    sim_days = st.slider("Historical days to simulate", min_value=7, max_value=90, value=30, step=1)
    overflow_threshold = st.slider("Overflow threshold (%)", 70, 100, 90, 1)
    auto_dispatch = st.checkbox("Auto-dispatch municipality for ALERT bins", value=False)
    st.markdown("---")
    st.caption("Use the 'Refresh' button to simulate new readings.")

# Ensure data
ensure_bins_master()
ensure_weather_profiles()
if not HISTORY_PATH.exists():
    with st.spinner("Generating synthetic history..."):
        simulate_history(days=sim_days)

# Train (or not)
if retrain or not Path("models/classifier_decision_tree.pkl").exists():
    with st.spinner("Training models..."):
        metrics = train_models()
        st.success("Models trained.")

# Load latest data & build a 'current snapshot' by taking last record per bin
hist = load_data()
# Only keep last timestamp per bin
idx = hist.groupby("bin_id")["ts"].idxmax()
current = hist.loc[idx].copy()
current["ts"] = pd.to_datetime(current["ts"])

# Predict
y_overflow, y_next = predict_overflow_and_fill(current)
current["pred_overflow"] = y_overflow
current["pred_fill_next"] = np.clip(y_next, 0, 100)

# Determine alert bins
current["status"] = np.where(current["fill_level"] >= overflow_threshold, "ALERT",
                      np.where(current["pred_overflow"] == 1, "RISK", "OK"))
current["alert_reason"] = np.where(current["status"] == "ALERT", "Threshold exceeded", 
                            np.where(current["status"] == "RISK", "Model predicts overflow", ""))

alert_bins = current[current["status"] == "ALERT"]
risk_bins = current[current["status"] == "RISK"]

# Top KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total bins", len(current))
kpi2.metric("ALERT bins (≥ threshold)", int((current["status"] == "ALERT").sum()))
kpi3.metric("RISK bins (predicted)", int((current["status"] == "RISK").sum()))
kpi4.metric("Avg fill level", f"{current['fill_level'].mean():.1f}%")

# Map & table
st.subheader("City Map")
try:
    st.map(current.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude","longitude"]])
except Exception as e:
    st.info("Map not available in this environment.")

st.subheader("Bin Status")
show_cols = ["bin_id","area_type","nearby_population","capacity_l","fill_level","pred_fill_next","status","alert_reason","lat","lon"]
st.dataframe(current[show_cols].sort_values(["status","fill_level"], ascending=[True, False]), use_container_width=True)

# Metrics section
with st.expander("Model Metrics"):
    import json, os
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        cm = np.array(metrics["confusion_matrix"])
        st.write("Confusion Matrix (Decision Tree Classifier)")
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha="center", va="center")
        st.pyplot(fig)
        st.json(metrics["classification_report"])
    else:
        st.info("Metrics not available yet.")

# Alert actions
st.subheader("Dispatch Actions")
colA, colB = st.columns(2)
with colA:
    if st.button("Send ALERT bins to Municipality"):
        sent = notify_municipality(alert_bins.to_dict(orient="records"))
        st.success(f"Queued {sent} bins for collection. See logs/dispatch_outbox.jsonl")

with colB:
    if st.button("Refresh (simulate new readings)"):
        # Append one new hour of readings and re-run
        from bin_data_generator import simulate_history
        simulate_history(days=1)  # one more day appends and extends last timestamp
        st.experimental_rerun()

if auto_dispatch and len(alert_bins) > 0:
    _ = notify_municipality(alert_bins.to_dict(orient="records"))

# Alerts log
st.subheader("Recent Alerts")
recent_alerts = current[current["status"].isin(["ALERT","RISK"])][["bin_id","fill_level","pred_fill_next","status","alert_reason","lat","lon"]]
st.dataframe(recent_alerts.sort_values(["status","fill_level"], ascending=[True, False]), use_container_width=True)

st.caption("Logs written to logs/alerts.jsonl and logs/dispatch_outbox.jsonl (mock integrations).")
