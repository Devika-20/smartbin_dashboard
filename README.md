# SmartBin Overflow Prediction – Streamlit Dashboard

A fully functional, **software-only** prototype for predicting urban public waste-bin overflow and alerting the municipality.
It uses **simulated data**, trains ML models (Decision Tree & Linear Regression), and provides a **Streamlit dashboard** for real-time monitoring and alerts.

## Features
- Simulated bin telemetry (fill level, time, weather, area type, population, etc.)
- Model training pipeline: Decision Tree classifier for **Overflow risk** and Linear Regression for **fill level forecasting**
- Real-time prediction loop (simulated), threshold-based alerts
- Interactive Streamlit dashboard:
  - Map & table of bins with live status
  - Confusion matrix, classification report, and model metrics
  - Manual **"Send to Municipality"** action + automatic alert logging
- Pluggable notifier with placeholders for SMTP/email, Twilio SMS, or webhook

## Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app**
   ```bash
   streamlit run app.py
   ```
3. The app will:
   - Generate synthetic data (if missing)
   - Train models (if missing or retraining requested)
   - Start a simulated stream and show alerts

## Project Structure
```
smartbin_dashboard/
├── app.py
├── bin_data_generator.py
├── model_training.py
├── inference.py
├── notifier.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
│   ├── bins_master.csv
│   ├── history.csv
│   └── weather_profiles.json
├── models/
│   ├── classifier_decision_tree.pkl
│   └── regressor_linear.pkl
└── logs/
    ├── alerts.jsonl
    └── dispatch_outbox.jsonl
```

## Notes
- This prototype follows the **software-only + simulated data** approach described in your PPT (no hardware required).
- You can tweak the generator distributions in `bin_data_generator.py` to mimic your city zones and seasons.
- For production, connect `notifier.py` to your municipality systems (SMTP, SMS, or REST webhook).
