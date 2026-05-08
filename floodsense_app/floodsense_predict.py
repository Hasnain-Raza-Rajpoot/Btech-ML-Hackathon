"""
FLOODSENSE PREDICT PIPELINE
============================
Production-ready prediction function for the Streamlit UI.
Phase D-2: now includes district context (population, last flood, 7-day trajectory).
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE        = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
MODELS_DIR  = os.path.join(HERE, "final_models")
LOOKUP_PATH = os.path.join(HERE, "floodsense_onset.csv")

TRAINED_DISTRICTS = {"Sindh_District", "Balochistan_District", "KP_District"}

DISTRICT_FALLBACK = {
    "Sindh_District":   "Sindh_District",
    "Karachi":          "Sindh_District",
    "Hyderabad":        "Sindh_District",
    "Larkana":          "Sindh_District",
    "Sukkur":           "Sindh_District",
    "Mirpur Khas":      "Sindh_District",
    "Thatta":           "Sindh_District",
    "Dadu":             "Sindh_District",
    "Sanghar":          "Sindh_District",
    "Balochistan_District": "Balochistan_District",
    "Quetta":           "Balochistan_District",
    "Lasbela":          "Balochistan_District",
    "Khuzdar":          "Balochistan_District",
    "Sibi":             "Balochistan_District",
    "Turbat":           "Balochistan_District",
    "Gwadar":           "Balochistan_District",
    "Kech":             "Balochistan_District",
    "KP_District":      "KP_District",
    "Peshawar":         "KP_District",
    "Nowshera":         "KP_District",
    "Buner":            "KP_District",
    "Swat":             "KP_District",
    "Charsadda":        "KP_District",
    "Mansehra":         "KP_District",
    "Mardan":           "KP_District",
    "Dera Ismail Khan": "KP_District",
    "Abbottabad":       "KP_District",
    "Kohat":            "KP_District",
    "Lahore":           "KP_District",
    "Faisalabad":       "Sindh_District",
    "Multan":           "Sindh_District",
    "Rawalpindi":       "KP_District",
    "Bahawalpur":       "Sindh_District",
    "Sialkot":          "KP_District",
    "Gujranwala":       "KP_District",
    "Dera Ghazi Khan":  "Sindh_District",
    "Gilgit":           "KP_District",
    "Skardu":           "KP_District",
    "Muzaffarabad":     "KP_District",
}

DISTRICT_DISPLAY = {
    "Sindh_District":        "Sindh District  (training data)",
    "Balochistan_District":  "Balochistan District  (training data)",
    "KP_District":           "KP District  (training data)",
    "Karachi":          "Karachi  (Sindh)",
    "Hyderabad":        "Hyderabad  (Sindh)",
    "Larkana":          "Larkana  (Sindh)",
    "Sukkur":           "Sukkur  (Sindh)",
    "Mirpur Khas":      "Mirpur Khas  (Sindh)",
    "Thatta":           "Thatta  (Sindh)",
    "Dadu":             "Dadu  (Sindh)",
    "Sanghar":          "Sanghar  (Sindh)",
    "Quetta":           "Quetta  (Balochistan)",
    "Lasbela":          "Lasbela  (Balochistan)",
    "Khuzdar":          "Khuzdar  (Balochistan)",
    "Sibi":             "Sibi  (Balochistan)",
    "Turbat":           "Turbat  (Balochistan)",
    "Gwadar":           "Gwadar  (Balochistan)",
    "Kech":             "Kech  (Balochistan)",
    "Peshawar":         "Peshawar  (KP)",
    "Nowshera":         "Nowshera  (KP)",
    "Buner":            "Buner  (KP)",
    "Swat":             "Swat  (KP)",
    "Charsadda":        "Charsadda  (KP)",
    "Mansehra":         "Mansehra  (KP)",
    "Mardan":           "Mardan  (KP)",
    "Dera Ismail Khan": "Dera Ismail Khan  (KP)",
    "Abbottabad":       "Abbottabad  (KP)",
    "Kohat":            "Kohat  (KP)",
    "Lahore":           "Lahore  (Punjab)",
    "Faisalabad":       "Faisalabad  (Punjab)",
    "Multan":           "Multan  (Punjab)",
    "Rawalpindi":       "Rawalpindi  (Punjab)",
    "Bahawalpur":       "Bahawalpur  (Punjab)",
    "Sialkot":          "Sialkot  (Punjab)",
    "Gujranwala":       "Gujranwala  (Punjab)",
    "Dera Ghazi Khan":  "Dera Ghazi Khan  (Punjab)",
    "Gilgit":           "Gilgit  (Gilgit-Baltistan)",
    "Skardu":           "Skardu  (Gilgit-Baltistan)",
    "Muzaffarabad":     "Muzaffarabad  (AJK)",
}

SOIL_MAP = {"Dry": 0.20, "Moist": 0.50, "Saturated": 0.85}


# -----------------------------------------------------------------------------
# DISTRICT POPULATION (approximate, latest census-era estimates)
# Used for "population at risk" estimation.
# -----------------------------------------------------------------------------
DISTRICT_POPULATIONS = {
    # Training districts — entire province aggregates
    "Sindh_District":        50_000_000,
    "Balochistan_District":  14_000_000,
    "KP_District":           40_000_000,
    # Sindh
    "Karachi": 16_000_000, "Hyderabad": 1_700_000, "Larkana": 1_500_000,
    "Sukkur": 500_000, "Mirpur Khas": 1_500_000, "Thatta": 1_700_000,
    "Dadu": 1_600_000, "Sanghar": 2_000_000,
    # Punjab
    "Lahore": 13_000_000, "Faisalabad": 3_500_000, "Multan": 2_000_000,
    "Rawalpindi": 2_100_000, "Bahawalpur": 750_000, "Sialkot": 656_000,
    "Gujranwala": 2_000_000, "Dera Ghazi Khan": 3_000_000,
    # KP
    "Peshawar": 2_200_000, "Nowshera": 1_500_000, "Buner": 900_000,
    "Swat": 2_300_000, "Charsadda": 1_600_000, "Mansehra": 1_500_000,
    "Mardan": 2_400_000, "Dera Ismail Khan": 1_600_000,
    "Abbottabad": 1_300_000, "Kohat": 990_000,
    # Balochistan
    "Quetta": 1_000_000, "Lasbela": 575_000, "Khuzdar": 800_000,
    "Sibi": 180_000, "Turbat": 750_000, "Gwadar": 263_000, "Kech": 750_000,
    # GB / AJK
    "Gilgit": 330_000, "Skardu": 270_000, "Muzaffarabad": 750_000,
}

# Risk-level multipliers — % of district population in flood-vulnerable areas
RISK_MULTIPLIERS = {
    "Low":      0.00,
    "Medium":   0.05,
    "High":     0.15,
    "Critical": 0.30,
}


# -----------------------------------------------------------------------------
# COMPREHENSIVE PLAIN-ENGLISH FEATURE LABELS
# -----------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    "precipitation":                "Today's rainfall",
    "soil_moisture":                "Soil moisture",
    "humidity":                     "Humidity",
    "temperature":                  "Temperature",
    "pressure":                     "Atmospheric pressure",
    "is_monsoon":                   "Monsoon season",
    "is_winter":                    "Winter season",
    "is_pre_monsoon":               "Pre-monsoon season",
    "is_post_monsoon":              "Post-monsoon season",
    "avg_elevation_m":              "District elevation",
    "month":                        "Month of year",
    "is_missing_precipitation":     "Rainfall data missing",
    "water_area_km2_lag1":          "Yesterday's water extent",
    "water_area_km2_lag2":          "Water extent 2 days ago",
    "water_area_km2_lag3":          "Water extent 3 days ago",
    "water_area_km2_lag5":          "Water extent 5 days ago",
    "water_area_km2_lag7":          "Water extent 7 days ago",
    "water_area_km2_lag14":         "Water extent 2 weeks ago",
    "precipitation_lag1":           "Yesterday's rainfall",
    "precipitation_lag2":           "Rainfall 2 days ago",
    "precipitation_lag3":           "Rainfall 3 days ago",
    "precipitation_lag5":           "Rainfall 5 days ago",
    "precipitation_lag7":           "Rainfall 7 days ago",
    "soil_moisture_lag1":           "Yesterday's soil moisture",
    "soil_moisture_lag2":           "Soil moisture 2 days ago",
    "soil_moisture_lag3":           "Soil moisture 3 days ago",
    "humidity_lag1":                "Yesterday's humidity",
    "humidity_lag2":                "Humidity 2 days ago",
    "pressure_lag1":                "Yesterday's pressure",
    "precipitation_3day_sum":       "3-day cumulative rainfall",
    "precipitation_7day_sum":       "7-day cumulative rainfall",
    "precipitation_14day_sum":      "14-day cumulative rainfall",
    "precipitation_30day_sum":      "30-day cumulative rainfall",
    "soil_moisture_7day_avg":       "7-day average soil moisture",
    "temperature_7day_avg":         "7-day average temperature",
    "precipitation_zscore":         "Rainfall anomaly (vs district norm)",
    "soil_moisture_zscore":         "Soil saturation anomaly",
    "temperature_zscore":           "Temperature anomaly",
    "pressure_zscore":              "Atmospheric pressure anomaly",
    "water_area_km2_lag1_zscore":   "Yesterday's water extent anomaly",
    "water_area_km2_lag2_zscore":   "Water extent 2 days ago (anomaly)",
    "water_area_km2_lag3_zscore":   "Water extent 3 days ago (anomaly)",
    "water_area_km2_lag5_zscore":   "Water extent 5 days ago (anomaly)",
    "water_area_km2_lag7_zscore":   "Water extent 7 days ago (anomaly)",
    "water_area_km2_lag14_zscore":  "Water extent 2 weeks ago (anomaly)",
    "precip_change_1day":           "Daily rainfall change",
    "soil_moisture_change_3day":    "Soil moisture change (3-day)",
    "pressure_drop_1day":           "Pressure drop (last 24h)",
    "days_since_heavy_rain":        "Days since last heavy rain",
    "days_since_dry_day":           "Days since last dry day",
    "days_in_monsoon":              "Days into monsoon season",
    "precip_x_soil_moisture":       "Rainfall × soil saturation",
    "precip_7day_x_monsoon":        "7-day rainfall × monsoon",
    "pressure_low_x_humidity_high": "Storm conditions index",
}


def get_risk_level(probability: float):
    if probability < 0.25:
        return "Low",      "🟢"
    elif probability < 0.50:
        return "Medium",   "🟡"
    elif probability < 0.75:
        return "High",     "🟠"
    else:
        return "Critical", "🔴"


ACTIONS = {
    "Low":      "No action needed. Continue routine monitoring.",
    "Medium":   "Issue community advisory. Brief local responders.",
    "High":     "Pre-position rescue resources. Alert vulnerable populations in low-lying areas.",
    "Critical": "Issue evacuation alert for low-lying areas. Activate emergency response.",
}


# -----------------------------------------------------------------------------
# LAZY-LOADED ARTIFACTS
# -----------------------------------------------------------------------------
_artifacts = {}

def _load_artifacts():
    if _artifacts:
        return _artifacts
    _artifacts["cont_model"]  = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgb_continuation_final.txt"))
    _artifacts["onset_model"] = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgb_onset_final.txt"))
    with open(os.path.join(MODELS_DIR, "shap_explainer_continuation.pkl"), "rb") as f:
        _artifacts["cont_explainer"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "shap_explainer_onset.pkl"), "rb") as f:
        _artifacts["onset_explainer"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
        _artifacts["feature_cols"] = pickle.load(f)

    lookup = pd.read_csv(LOOKUP_PATH)
    lookup["date"] = pd.to_datetime(lookup["date"])
    _artifacts["lookup"] = lookup

    # Phase D-2: load OOF onset predictions for trajectory computation
    oof_path = os.path.join(MODELS_DIR, "oof_preds_onset.npy")
    _artifacts["oof_preds_onset"] = np.load(oof_path) if os.path.exists(oof_path) else None
    return _artifacts


# -----------------------------------------------------------------------------
# PHASE D-2 HELPERS
# -----------------------------------------------------------------------------
def _format_population(p):
    """Format population: 1500000 → '1.5M', 750000 → '750K'."""
    if p is None or p == 0:
        return "—"
    if p >= 1_000_000:
        return f"{p / 1_000_000:.1f}M"
    if p >= 1_000:
        return f"{p / 1_000:.0f}K"
    return f"{p:,}"


def _estimate_population_at_risk(district, risk_level):
    """Returns (at_risk_count, total_population)."""
    pop = DISTRICT_POPULATIONS.get(district)
    if pop is None:
        # Default for unmapped districts — use 1M as reasonable estimate
        pop = 1_000_000
    multiplier = RISK_MULTIPLIERS.get(risk_level, 0.0)
    return int(pop * multiplier), pop


def _last_major_flood(effective_district, lookup_df):
    """Most recent flood_event=1 date for the effective district."""
    floods = lookup_df[(lookup_df["district"] == effective_district) &
                        (lookup_df["flood_event"] == 1)]
    if len(floods) == 0:
        return None
    return floods["date"].max()


def _compute_trajectory(effective_district, end_date, lookup_df, oof_preds_onset):
    """7-day risk trajectory from OOF predictions for the effective district."""
    if oof_preds_onset is None:
        return None

    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.Timedelta(days=7)
    sub = lookup_df[(lookup_df["district"] == effective_district) &
                     (lookup_df["date"] >= start_date) &
                     (lookup_df["date"] <= end_date)].sort_values("date")
    sub = sub.drop_duplicates(subset="date", keep="first")

    # Fall back to most recent 7 days if window has no data (e.g., future date)
    if len(sub) < 3:
        sub = (lookup_df[lookup_df["district"] == effective_district]
               .sort_values("date").tail(7).drop_duplicates(subset="date", keep="first"))

    if len(sub) < 3:
        return None

    onset_probs = oof_preds_onset[sub.index.values]
    onset_probs = np.where(np.isnan(onset_probs), 0, onset_probs)

    mid = len(onset_probs) // 2
    first_half_avg  = float(np.mean(onset_probs[:mid])) if mid > 0 else float(onset_probs[0])
    second_half_avg = float(np.mean(onset_probs[mid:]))
    diff = second_half_avg - first_half_avg

    if diff > 0.10:
        return {"trend": "rising",  "icon": "↗", "label": "Rising",  "delta": diff}
    elif diff < -0.10:
        return {"trend": "falling", "icon": "↘", "label": "Falling", "delta": diff}
    else:
        return {"trend": "stable",  "icon": "→", "label": "Stable",  "delta": diff}


# -----------------------------------------------------------------------------
# CORE FEATURE LOOKUP + USER OVERRIDES
# -----------------------------------------------------------------------------
def _apply_user_signal_overrides(row, rainfall, soil_condition, water_visible,
                                   lookup, effective_district):
    row["precipitation"] = rainfall
    if "soil_moisture" in row.columns:
        row["soil_moisture"] = SOIL_MAP.get(soil_condition, 0.5)

    if water_visible:
        flood_rows = lookup[(lookup["district"] == effective_district) &
                             (lookup["flood_event"] == 1)]
        if len(flood_rows) > 0:
            for col in ["water_area_km2_lag1", "water_area_km2_lag2", "water_area_km2_lag3"]:
                if col in row.columns and col in flood_rows.columns:
                    val = flood_rows[col].median()
                    if pd.notna(val):
                        row[col] = val

    sustained = (water_visible and soil_condition == "Saturated" and rainfall >= 30)
    if sustained:
        flood_rows = lookup[(lookup["district"] == effective_district) &
                             (lookup["flood_event"] == 1)]
        if len(flood_rows) > 0:
            for col in ["water_area_km2_lag5", "water_area_km2_lag7", "water_area_km2_lag14"]:
                if col in row.columns and col in flood_rows.columns:
                    val = flood_rows[col].median()
                    if pd.notna(val):
                        row[col] = val

    district_data = lookup[lookup["district"] == effective_district]
    for base in ["water_area_km2_lag1", "water_area_km2_lag2", "water_area_km2_lag3",
                 "water_area_km2_lag5", "water_area_km2_lag7", "water_area_km2_lag14"]:
        z_col = base + "_zscore"
        if z_col in row.columns and base in row.columns and base in district_data.columns:
            mean = district_data[base].mean()
            std  = district_data[base].std()
            if std and std > 0:
                row[z_col] = (row[base] - mean) / std
    return row


def _get_features(date, district, rainfall_today, soil_condition, water_visible):
    art = _load_artifacts()
    lookup = art["lookup"]
    feature_cols = art["feature_cols"]

    date = pd.to_datetime(date)
    is_unknown = district not in TRAINED_DISTRICTS
    effective = DISTRICT_FALLBACK.get(district, "KP_District") if is_unknown else district

    matches = lookup[(lookup["district"] == effective) & (lookup["date"] == date)]
    used_date_fallback = False
    fallback_direction = None

    if len(matches) == 0:
        # 1. Try the most recent date BEFORE the requested date (preferred)
        prior = lookup[(lookup["district"] == effective) & (lookup["date"] < date)]
        if len(prior) > 0:
            row = prior.sort_values("date").iloc[-1:].copy()
            used_date_fallback = True
            fallback_direction = "backward"
        else:
            # 2. No prior data — fall back FORWARD to the earliest available date
            #    (happens when user picks a date before the district's data starts —
            #    e.g. Balochistan + Jan 2022 when Balochistan data starts 2023)
            forward = lookup[lookup["district"] == effective]
            if len(forward) == 0:
                raise ValueError(f"No data at all for {effective}")
            row = forward.sort_values("date").head(1).copy()
            used_date_fallback = True
            fallback_direction = "forward"
    else:
        row = matches.iloc[:1].copy()

    row = _apply_user_signal_overrides(
        row, rainfall_today, soil_condition, water_visible, lookup, effective,
    )

    X = row[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    return X, {
        "is_unknown_district":   is_unknown,
        "effective_district":    effective,
        "used_date_fallback":    used_date_fallback,
        "fallback_direction":    fallback_direction,
        "lookup_date":           row["date"].iloc[0],
    }


def _explain(model_name, X_row, top_k=5):
    art = _load_artifacts()
    explainer    = art[f"{model_name}_explainer"]
    feature_cols = art["feature_cols"]
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    drivers = []
    for idx in order:
        feat = feature_cols[idx]
        label = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title())
        value = X_row.iloc[0][feat]
        contribution = sv[idx]
        sign = "↑" if contribution > 0 else "↓"
        drivers.append({
            "feature": feat, "label": label, "value": float(value),
            "shap": float(contribution), "sign": sign,
            "text": f"{sign} {label}: {value:.2f}",
        })
    return drivers


# -----------------------------------------------------------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_flood_risk(date, district, rainfall_today, soil_condition,
                        water_visible, threshold=0.3):
    art = _load_artifacts()
    X, meta = _get_features(date, district, rainfall_today, soil_condition, water_visible)
    p_onset = float(art["onset_model"].predict(X.values)[0])
    p_cont  = float(art["cont_model"].predict(X.values)[0])

    onset_fires = p_onset >= threshold
    cont_fires  = p_cont  >= threshold
    if onset_fires and cont_fires:    flood_state = "new_onset"
    elif cont_fires:                  flood_state = "ongoing"
    elif onset_fires:                 flood_state = "rising_risk"
    else:                              flood_state = "no_flood"

    primary_prob = max(p_onset, p_cont)
    risk_level, indicator = get_risk_level(primary_prob)

    if p_onset >= p_cont:
        drivers = _explain("onset", X, top_k=5)
    else:
        drivers = _explain("cont",  X, top_k=5)

    raw_confidence = primary_prob * 100
    if meta["is_unknown_district"]:
        confidence = round(raw_confidence * 0.7)
        warning = (f"District '{district}' is not in the model's training data. "
                   f"Predictions use {meta['effective_district']} as a regional analogue. "
                   f"Confidence has been reduced by 30%. Manual assessment by local "
                   f"authorities is strongly recommended.")
    else:
        confidence = round(raw_confidence)
        warning = None

    if meta["used_date_fallback"]:
        if meta.get("fallback_direction") == "forward":
            date_warning = (f"No data available before requested date for "
                            f"{meta['effective_district']}. Using earliest available data "
                            f"({meta['lookup_date'].date()}) as baseline.")
        else:
            date_warning = (f"Exact date not in lookup table. Using most recent prior data "
                            f"({meta['lookup_date'].date()}).")
        warning = (warning + " " + date_warning) if warning else date_warning

    # ── Phase D-2: district context ─────────────────────────────────────────
    pop_at_risk, total_pop = _estimate_population_at_risk(district, risk_level)
    last_flood = _last_major_flood(meta["effective_district"], art["lookup"])
    trajectory = _compute_trajectory(
        meta["effective_district"], date, art["lookup"], art["oof_preds_onset"],
    )

    return {
        "risk_level":               risk_level,
        "indicator":                indicator,
        "confidence":               confidence,
        "flood_state":              flood_state,
        "explanation":              drivers,
        "explanation_text":         [d["text"] for d in drivers],
        "recommended_action":       ACTIONS[risk_level],
        "warning":                  warning,
        "onset_probability":        round(p_onset, 4),
        "continuation_probability": round(p_cont, 4),
        "effective_district":       meta["effective_district"],
        "is_unknown_district":      meta["is_unknown_district"],
        "lookup_date":              meta["lookup_date"],
        # Phase D-2 fields
        "district_population":      total_pop,
        "district_population_fmt":  _format_population(total_pop),
        "population_at_risk":       pop_at_risk,
        "population_at_risk_fmt":   _format_population(pop_at_risk),
        "last_major_flood":         last_flood,
        "risk_trajectory":          trajectory,
    }