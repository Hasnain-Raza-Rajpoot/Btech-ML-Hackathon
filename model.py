import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

train = pd.read_csv('floodsense_engineered.csv')

train['precip_cumulative']    = train['precip_3day_avg'] + train['precip_7day_avg']
train['soil_saturation_risk'] = train['soil_moisture'] * train['precip_3day_avg']
train['monsoon_x_precip']     = train['is_monsoon'] * train['precipitation']
train['monsoon_x_soil']       = train['is_monsoon'] * train['soil_moisture']
train['elevation_risk']       = 1 / (train['avg_elevation_m'] + 1)
train['temp_humidity_index']  = train['temperature'] * train['humidity'] / 100
train['precip_acceleration']  = train['precip_3day_avg'] - train['precip_7day_avg']
train['soil_lag2']            = train.groupby('district')['soil_moisture'].shift(2).fillna(0)
train['precip_lag7']          = train.groupby('district')['precipitation'].shift(7).fillna(0)

drop_cols = ['date', 'district', 'flood_event',
             'water_area_km2', 'water_area_change',
             'water_area_pct_change', 'ds_idx',
             'soil_x_water', 'water_lag1', 'rain_x_elevation' , 'year']

X = train.drop(columns=drop_cols)
y = train['flood_event']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train_bal, y_train_bal)
y_pred_dt = dt_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_bal, y_train_bal)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

feature_importance_xgb = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("Top 10 Features:")
print(feature_importance_xgb.head(10))

print(f"\nComparison:")
print(f"  Decision Tree : {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"  Random Forest : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  XGBoost       : {accuracy_score(y_test, y_pred_xgb):.4f}")

def get_risk_level(probability):
    if probability < 0.25:
        return "Low", "🟢"
    elif probability < 0.50:
        return "Medium", "🟡"
    elif probability < 0.75:
        return "High", "🟠"
    else:
        return "Critical", "🔴"

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    'actual'      : y_test.values,
    'probability' : y_prob_xgb,
    'confidence'  : (y_prob_xgb * 100).round(1),
})

results['risk_level'] = results['probability'].apply(lambda p: get_risk_level(p)[0])
results['indicator']  = results['probability'].apply(lambda p: get_risk_level(p)[1])

print("\nRisk Level Distribution:")
print(results['risk_level'].value_counts())

print("\nSample Predictions:")
print(results[['actual','confidence','risk_level','indicator']].head(15).to_string())

print("\nRisk level vs actual flood:")
print(pd.crosstab(results['risk_level'], results['actual'],
      rownames=['Risk Level'], colnames=['Actual Flood']))




joblib.dump(xgb_model, 'floodsense_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_features.pkl')

print("Model saved to floodsense_model.pkl")
print("Features saved to model_features.pkl")






#
# train_full = train.copy()
# train_full['prob'] = xgb_model.predict_proba(X)[:, 1]
# train_full['confidence'] = (train_full['prob'] * 100).round(1)
# train_full['risk'] = train_full['prob'].apply(lambda p: get_risk_level(p)[0])
#
# print("\n" + "="*60)
# print("REAL FLOOD EVENTS FLAGGED HIGH/CRITICAL")
# print("="*60)
#
# flood_flagged = train_full[
#     (train_full['flood_event'] == 1) &
#     (train_full['risk'].isin(['High', 'Critical']))
# ]
#
# print(flood_flagged[['date','district','precipitation',
#                       'soil_moisture','confidence','risk']].head(10).to_string())
#
# print(f"\nCorrectly flagged: {len(flood_flagged)}")
# print(f"Total flood events: {(train_full['flood_event']==1).sum()}")
#
# print("\n" + "="*60)
# print("HIGHEST RAINFALL FLOOD EVENTS IN SINDH")
# print("="*60)
#
# sindh = train_full[
#     (train_full['district'] == 'Sindh_District') &
#     (train_full['flood_event'] == 1)
# ].sort_values('precipitation', ascending=False)
#
# print(sindh[['date','precipitation','soil_moisture',
#              'confidence','risk']].head(10).to_string())