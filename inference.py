from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

CLF_PATH = MODELS_DIR / "classifier_decision_tree.pkl"
REG_PATH = MODELS_DIR / "regressor_linear.pkl"

def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg

FEATURES = ["hour","dow","temp_c","festival","nearby_population","capacity_l","fill_level"]

def predict_overflow_and_fill(df_current: pd.DataFrame):
    clf, reg = load_models()
    X_cls = df_current[FEATURES]
    y_overflow = clf.predict(X_cls)

    X_reg = df_current[["hour","dow","temp_c","festival","nearby_population","capacity_l","fill_level"]]
    y_next = reg.predict(X_reg)
    return y_overflow.astype(int), y_next
