"""
train_for_leaderboard.py — Trains a sklearn-compatible LightGBM model
on the 17 features that the BTech leaderboard's run_prediction.py expects.

Run this BEFORE run_prediction.py.
Output: model.pkl
"""
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Load cleaned data — adjust path if needed
DATA_PATH = "processed/floodsense_clean.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# Encode district as ds_idx (matches what the leaderboard SCENARIOS use)
DS_IDX = {
    "Sindh_District":       2.0,
    "KP_District":          1.0,
    "Balochistan_District": 4.0,
}
df["ds_idx"] = df["district"].map(DS_IDX)

# 17 features the leaderboard script expects
FEATURE_COLS = [
    "precipitation", "precip_3day_avg", "precip_7day_avg",
    "soil_moisture", "soil_3day_avg",
    "water_area_km2", "water_area_change", "water_area_pct_change",
    "temperature", "humidity", "pressure", "evaporation", "wind_speed",
    "month", "day_of_year", "is_monsoon", "ds_idx",
]
X = df[FEATURE_COLS].copy()
y = df["flood_event"].values

# Drop rows with NaN (water_area_change has NaN at first row of each district)
mask = X.notna().all(axis=1)
X = X[mask].reset_index(drop=True)
y = y[mask]
print(f"After dropping NaN: {X.shape}, {y.sum()} positives ({y.mean():.1%})")

# Train LightGBM with sklearn API for predict_proba compatibility
model = LGBMClassifier(
    objective="binary",
    n_estimators=400,
    learning_rate=0.01,
    num_leaves=32,
    min_data_in_leaf=15,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
)
model.fit(X, y)

probs = model.predict_proba(X)[:, 1]
print(f"In-sample AUC: {roc_auc_score(y, probs):.4f}")

joblib.dump(model, "final_models/sklearn_style_model.pkl")
print("Saved: model.pkl")