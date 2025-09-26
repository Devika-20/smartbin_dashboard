from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    hist = pd.read_csv(DATA_DIR / "history.csv", parse_dates=["ts"])
    return hist

def feature_target(df: pd.DataFrame):
    # Targets:
    # (1) overflow now (classification)
    y_class = df["overflow"].astype(int)

    # (2) next-hour fill level (regression): create lagged features and lead target
    df_sorted = df.sort_values(["bin_id", "ts"])
    df_sorted["fill_next"] = df_sorted.groupby("bin_id")["fill_level"].shift(-1)
    df_reg = df_sorted.dropna(subset=["fill_next"])

    feature_cols = ["hour","dow","temp_c","festival","nearby_population","capacity_l"]
    X_class = df[feature_cols + ["fill_level"]]
    X_reg = df_reg[feature_cols + ["fill_level"]]
    y_reg = df_reg["fill_next"]
    return X_class, y_class, X_reg, y_reg

def train_models(random_state=42):
    df = load_data()
    Xc, yc, Xr, yr = feature_target(df)

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=random_state, stratify=yc)
    clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=random_state)
    clf.fit(Xc_tr, yc_tr)

    y_pred = clf.predict(Xc_te)
    cm = confusion_matrix(yc_te, y_pred)
    cr = classification_report(yc_te, y_pred, output_dict=True)

    reg = LinearRegression()
    reg.fit(Xr, yr)

    joblib.dump(clf, MODELS_DIR / "classifier_decision_tree.pkl")
    joblib.dump(reg, MODELS_DIR / "regressor_linear.pkl")

    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "samples": {
            "train_cls": int(len(Xc_tr)),
            "test_cls": int(len(Xc_te)),
            "reg": int(len(Xr))
        }
    }
    (MODELS_DIR / "metrics.json").write_text(pd.Series(metrics).to_json())
    return metrics

if __name__ == "__main__":
    m = train_models()
    print("Models trained. Metrics keys:", list(m.keys()))
