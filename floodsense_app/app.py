"""
FloodSense — Streamlit UI (Phase D-2)
=======================================
Adds district context card with population at risk, last major flood,
and 7-day risk trajectory.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import date
from floodsense_predict import (
    predict_flood_risk,
    DISTRICT_FALLBACK,
    DISTRICT_DISPLAY,
)
import floodsense_charts as fc


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodSense — AI Flood Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# CUSTOM CSS — Pakistani green theme + Phase D-2 context grid
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ==========  BASE TYPOGRAPHY & SPACING  ========== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}
h1 {
    color: #01411C !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stCaptionContainer"] {
    color: #6B7280 !important;
    font-size: 1rem !important;
    margin-bottom: 1.5rem !important;
}
h3 {
    color: #1F2937 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #01411C20;
    padding-bottom: 0.5rem;
    margin-top: 0.5rem !important;
}
hr {
    border-color: #01411C20 !important;
    margin: 1.5rem 0 !important;
}

/* ==========  RISK BANNER  ========== */
.risk-banner {
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    border-left: 10px solid;
}
.risk-banner-left   { display: flex; align-items: center; gap: 24px; }
.risk-banner-emoji  { font-size: 56px; line-height: 1; }
.risk-banner-text   { display: flex; flex-direction: column; gap: 4px; }
.risk-banner-label  { font-size: 36px; font-weight: 800; text-transform: uppercase;
                       letter-spacing: -0.5px; line-height: 1; }
.risk-banner-status { font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px;
                       font-weight: 600; opacity: 0.7; }
.risk-banner-right        { text-align: right; }
.risk-banner-conf-num     { font-size: 48px; font-weight: 800; line-height: 1; }
.risk-banner-conf-label   { font-size: 11px; text-transform: uppercase; letter-spacing: 2px;
                             font-weight: 600; opacity: 0.7; margin-top: 4px; }
.risk-low      { background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
                  color: #14532D; border-left-color: #16A34A; }
.risk-medium   { background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
                  color: #78350F; border-left-color: #EAB308; }
.risk-high     { background: linear-gradient(135deg, #FFEDD5 0%, #FED7AA 100%);
                  color: #7C2D12; border-left-color: #EA580C; }
.risk-critical { background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
                  color: #7F1D1D; border-left-color: #DC2626; }

/* ==========  OUTPUT SECTION CARDS  ========== */
.output-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #E5E7EB;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.output-card-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6B7280;
    margin-bottom: 12px;
}
.output-card-action {
    font-size: 16px;
    font-weight: 600;
    line-height: 1.5;
}
.output-card-action.action-low      { color: #14532D; }
.output-card-action.action-medium   { color: #78350F; }
.output-card-action.action-high     { color: #7C2D12; }
.output-card-action.action-critical { color: #7F1D1D; }

.driver-list { list-style: none; padding: 0; margin: 0; }
.driver-item {
    padding: 10px 14px;
    margin: 6px 0;
    background: #F9FAFB;
    border-radius: 8px;
    border-left: 3px solid #01411C40;
    font-size: 14px;
    line-height: 1.4;
}
.driver-item.driver-up   { border-left-color: #DC2626; }
.driver-item.driver-down { border-left-color: #16A34A; }

.prob-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.prob-card {
    background: #F9FAFB;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #E5E7EB;
}
.prob-card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    color: #6B7280;
}
.prob-card-value {
    font-size: 28px;
    font-weight: 800;
    color: #01411C;
    margin-top: 4px;
    line-height: 1;
}

/* ==========  PHASE D-2: DISTRICT CONTEXT GRID  ========== */
.context-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.context-item {
    text-align: center;
    padding: 14px 10px;
    background: #F9FAFB;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}
.context-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    color: #6B7280;
}
.context-value {
    font-size: 22px;
    font-weight: 800;
    color: #01411C;
    margin-top: 6px;
    line-height: 1;
}
.context-sublabel {
    font-size: 10px;
    color: #9CA3AF;
    margin-top: 4px;
    font-weight: 500;
}
.context-trend-rising  { color: #DC2626; }
.context-trend-falling { color: #16A34A; }
.context-trend-stable  { color: #6B7280; }
.context-at-risk-high  { color: #DC2626; }
.context-at-risk-low   { color: #16A34A; }

/* ==========  ALERT MODE BANNER (Scenario 3 bonus)  ========== */
.alert-mode-banner {
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 20px;
    color: white;
    border-left: 8px solid white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    animation: alertPulse 2s ease-in-out infinite;
}
.alert-mode-critical { background: linear-gradient(135deg, #DC2626, #991B1B); }
.alert-mode-high     { background: linear-gradient(135deg, #EA580C, #9A3412); }
.alert-mode-header {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    opacity: 0.95;
    margin-bottom: 8px;
}
.alert-mode-title {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 14px;
}
.alert-mode-action {
    font-size: 15px;
    font-weight: 600;
    line-height: 1.5;
    background: rgba(255,255,255,0.18);
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 4px solid white;
}
@keyframes alertPulse {
    0%, 100% { box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
    50%      { box-shadow: 0 6px 28px rgba(220,38,38,0.45); }
}

/* ==========  WARNING & INPUT POLISH  ========== */
.warning-banner {
    background: #FEF3C7;
    border-left: 4px solid #EAB308;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 13px;
    color: #78350F;
    line-height: 1.5;
}

[data-testid="stTextInput"] > label,
[data-testid="stNumberInput"] > label,
[data-testid="stDateInput"] > label,
[data-testid="stSelectbox"] > label,
[data-testid="stRadio"] > label {
    font-weight: 600 !important;
    color: #374151 !important;
}

.stButton > button[kind="primary"] {
    background-color: #01411C !important;
    border-color: #01411C !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    height: 3rem !important;
    box-shadow: 0 4px 12px rgba(1, 65, 28, 0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2D6A4F !important;
    border-color: #2D6A4F !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(1, 65, 28, 0.35) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    border: 2px solid #E5E7EB !important;
    font-weight: 600 !important;
    color: #1F2937 !important;
    transition: all 0.2s ease !important;
    height: auto !important;
    padding: 12px 16px !important;
    line-height: 1.3 !important;
    text-align: left !important;
    white-space: normal !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #01411C !important;
    background: #F0FDF4 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(1, 65, 28, 0.1) !important;
}
[data-testid="stAlert"] { border-radius: 12px !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# WARM-UP
# -----------------------------------------------------------------------------
@st.cache_resource
def _warmup():
    from floodsense_predict import _load_artifacts
    return _load_artifacts()

_warmup()


@st.cache_data
def _load_chart_data():
    here = os.path.dirname(os.path.abspath(__file__))
    lookup = pd.read_csv(os.path.join(here, "floodsense_onset.csv"))
    lookup["date"] = pd.to_datetime(lookup["date"])
    oof_path = os.path.join(here, "final_models", "oof_preds_onset.npy")
    oof_preds_onset = np.load(oof_path) if os.path.exists(oof_path) else None
    return lookup, oof_preds_onset

LOOKUP_DF, OOF_PREDS_ONSET = _load_chart_data()


# -----------------------------------------------------------------------------
# DROPDOWN OPTIONS
# -----------------------------------------------------------------------------
PROVINCE_ORDER = {
    "training":     ["Sindh_District", "Balochistan_District", "KP_District"],
    "Sindh":        ["Karachi", "Hyderabad", "Larkana", "Sukkur", "Mirpur Khas",
                     "Thatta", "Dadu", "Sanghar"],
    "Punjab":       ["Lahore", "Faisalabad", "Multan", "Rawalpindi", "Bahawalpur",
                     "Sialkot", "Gujranwala", "Dera Ghazi Khan"],
    "KP":           ["Peshawar", "Nowshera", "Buner", "Swat", "Charsadda",
                     "Mansehra", "Mardan", "Dera Ismail Khan", "Abbottabad", "Kohat"],
    "Balochistan":  ["Quetta", "Lasbela", "Khuzdar", "Sibi", "Turbat", "Gwadar", "Kech"],
    "Other":        ["Gilgit", "Skardu", "Muzaffarabad"],
}
DISTRICT_LIST = (
    PROVINCE_ORDER["training"] + PROVINCE_ORDER["Sindh"] + PROVINCE_ORDER["Punjab"]
    + PROVINCE_ORDER["KP"] + PROVINCE_ORDER["Balochistan"] + PROVINCE_ORDER["Other"]
)


# -----------------------------------------------------------------------------
# DEMO SCENARIOS
# -----------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "🔴 Sindh Sept 2022 (catastrophic monsoon)": {
        "date": date(2022, 9, 8), "district": "Sindh_District",
        "rainfall": 85.0, "soil": "Saturated", "water_visible": True,
    },
    "🟠 KP June 2024 (regional flooding)": {
        "date": date(2024, 6, 29), "district": "KP_District",
        "rainfall": 70.0, "soil": "Saturated", "water_visible": True,
    },
    "🟡 Nowshera deployment scenario": {
        "date": date(2024, 7, 15), "district": "Nowshera",
        "rainfall": 60.0, "soil": "Saturated", "water_visible": True,
    },
    "🟢 Normal day (no flood signal)": {
        "date": date(2024, 2, 10), "district": "KP_District",
        "rainfall": 2.0, "soil": "Dry", "water_visible": False,
    },
}


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "app_initialized" not in st.session_state:
    st.session_state.in_date     = date(2024, 8, 15)
    st.session_state.in_district = "Nowshera"
    st.session_state.in_rainfall = 10.0
    st.session_state.in_soil     = "Moist"
    st.session_state.in_water    = False
    st.session_state.app_initialized = True


def apply_scenario(scenario):
    st.session_state.in_date     = scenario["date"]
    st.session_state.in_district = scenario["district"]
    st.session_state.in_rainfall = scenario["rainfall"]
    st.session_state.in_soil     = scenario["soil"]
    st.session_state.in_water    = scenario["water_visible"]


def render_alert_mode_banner(result, selected_district):
    """Scenario 3 bonus: Alert Mode for High/Critical districts.

    Activates automatically. Visible above the fold. Plain-language action
    for non-technical PDMA officials. Lightweight (no images) for slow internet.
    """
    risk_level = result["risk_level"]
    if risk_level not in ("High", "Critical"):
        return  # Only show for High/Critical, per scenario spec

    # Plain-language one-line recommended action for non-technical users
    ALERT_ACTIONS = {
        "Critical": ("IMMEDIATE EVACUATION REQUIRED. Deploy rescue teams. "
                     "Activate provincial emergency response. Open all designated shelters."),
        "High":     ("Issue evacuation advisory for low-lying areas. "
                     "Open emergency shelters. Pre-position relief supplies. Increase monitoring frequency."),
    }

    district_label = DISTRICT_DISPLAY.get(selected_district, selected_district)
    district_label = district_label.split("  ")[0]  # strip "(training data)" suffix

    severity_class = f"alert-mode-{risk_level.lower()}"

    html = f"""
    <div class="alert-mode-banner {severity_class}">
        <div class="alert-mode-header">🚨 Alert Mode Activated</div>
        <div class="alert-mode-title">{district_label} — {risk_level.upper()} RISK</div>
        <div class="alert-mode-action">
            <strong>Recommended Action:</strong> {ALERT_ACTIONS[risk_level]}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_compact_alert_card(alert):
    """Compact alert card for the system-wide alert dashboard.
    Shows district name, risk level, and one-line recommended action."""

    risk_level = alert["risk_level"]
    ALERT_ACTIONS = {
        "Critical": "IMMEDIATE EVACUATION. Deploy rescue teams. Open all designated shelters. Activate provincial emergency response.",
        "High":     "Issue evacuation advisory for low-lying areas. Open emergency shelters. Pre-position relief supplies.",
    }

    district_label = DISTRICT_DISPLAY.get(alert["district"], alert["district"]).split("  ")[0]
    severity_class = f"alert-mode-{risk_level.lower()}"

    html = f"""
    <div class="alert-mode-banner {severity_class}" style="margin-bottom: 12px; padding: 18px 22px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
            <div style="flex: 1;">
                <div class="alert-mode-header" style="margin-bottom: 4px;">
                    🚨 {risk_level.upper()} RISK
                </div>
                <div style="font-size: 22px; font-weight: 800; line-height: 1.1; margin-bottom: 10px;">
                    {district_label}
                </div>
                <div class="alert-mode-action" style="font-size: 14px;">
                    <strong>Action:</strong> {ALERT_ACTIONS[risk_level]}
                </div>
            </div>
            <div style="text-align: right; min-width: 90px;">
                <div style="font-size: 32px; font-weight: 800; line-height: 1;">{alert['confidence']}%</div>
                <div style="font-size: 10px; opacity: 0.85; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 4px;">
                    Confidence
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# OUTPUT RENDERERS
# -----------------------------------------------------------------------------
def render_risk_banner(result):
    risk_class = "risk-" + result["risk_level"].lower()
    flood_state_label = result["flood_state"].replace("_", " ").title()
    html = f"""
    <div class="risk-banner {risk_class}">
        <div class="risk-banner-left">
            <div class="risk-banner-emoji">{result['indicator']}</div>
            <div class="risk-banner-text">
                <div class="risk-banner-label">{result['risk_level']} Risk</div>
                <div class="risk-banner-status">Status: {flood_state_label}</div>
            </div>
        </div>
        <div class="risk-banner-right">
            <div class="risk-banner-conf-num">{result['confidence']}%</div>
            <div class="risk-banner-conf-label">Confidence</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_district_context_card(result):
    """Phase D-2: 4-metric grid showing district context."""
    pop_fmt        = result["district_population_fmt"]
    at_risk_fmt    = result["population_at_risk_fmt"]
    last_flood     = result.get("last_major_flood")
    trajectory     = result.get("risk_trajectory")

    last_flood_str = last_flood.strftime("%b %Y") if last_flood is not None else "—"

    if trajectory:
        traj_value = f"{trajectory['icon']} {trajectory['label']}"
        traj_class = f"context-trend-{trajectory['trend']}"
        delta_pct  = trajectory.get("delta", 0) * 100
        traj_sub   = f"{delta_pct:+.0f} pts vs prior days"
    else:
        traj_value = "—"
        traj_class = ""
        traj_sub   = "Insufficient data"

    # Risk-color the at-risk number for emphasis
    at_risk_class = ""
    if result["risk_level"] in ("High", "Critical"):
        at_risk_class = "context-at-risk-high"
    elif result["risk_level"] == "Low":
        at_risk_class = "context-at-risk-low"

    html = f"""
    <div class="output-card">
        <div class="output-card-title">📊 District Context</div>
        <div class="context-grid">
            <div class="context-item">
                <div class="context-label">Population</div>
                <div class="context-value">{pop_fmt}</div>
                <div class="context-sublabel">district total</div>
            </div>
            <div class="context-item">
                <div class="context-label">Estimated at risk</div>
                <div class="context-value {at_risk_class}">{at_risk_fmt}</div>
                <div class="context-sublabel">in flood-prone areas</div>
            </div>
            <div class="context-item">
                <div class="context-label">Last major flood</div>
                <div class="context-value">{last_flood_str}</div>
                <div class="context-sublabel">in our records</div>
            </div>
            <div class="context-item">
                <div class="context-label">7-day trajectory</div>
                <div class="context-value {traj_class}">{traj_value}</div>
                <div class="context-sublabel">{traj_sub}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_action_card(result):
    risk_class = "action-" + result["risk_level"].lower()
    html = f"""
    <div class="output-card">
        <div class="output-card-title">📌 Recommended Action</div>
        <div class="output-card-action {risk_class}">{result['recommended_action']}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_drivers_card(result):
    drivers_html = ""
    for d in result["explanation"]:
        direction_class = "driver-up" if d["sign"] == "↑" else "driver-down"
        drivers_html += (
            f'<li class="driver-item {direction_class}">'
            f'<strong>{d["sign"]}</strong> &nbsp; {d["label"]}: '
            f'<strong>{d["value"]:.2f}</strong>'
            f'</li>'
        )
    html = f"""
    <div class="output-card">
        <div class="output-card-title">🔍 Why This Prediction?</div>
        <ul class="driver-list">{drivers_html}</ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_probs_card(result):
    html = f"""
    <div class="output-card">
        <div class="output-card-title">📊 Model Outputs</div>
        <div class="prob-grid">
            <div class="prob-card">
                <div class="prob-card-label">Onset Probability</div>
                <div class="prob-card-value">{result['onset_probability']:.1%}</div>
            </div>
            <div class="prob-card">
                <div class="prob-card-label">Continuation Probability</div>
                <div class="prob-card-value">{result['continuation_probability']:.1%}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_warning(result):
    if result["warning"]:
        html = f"""
        <div class="warning-banner">
            <strong>⚠️ Note:</strong> {result['warning']}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🌊 FloodSense")
st.caption("AI-powered flood early-warning system for Pakistan")

# -----------------------------------------------------------------------------
# SCENARIO 3 BONUS — SYSTEM-WIDE ALERT MODE DASHBOARD
# Activates automatically on page load. Shows all districts at High/Critical
# risk with plain-language recommended actions. Visible without scrolling.
# -----------------------------------------------------------------------------

# Districts being live-monitored (mix of provinces, mix of dates from real flood events)
LIVE_MONITORING = [
    {"district": "Sindh_District",       "date": date(2022, 9, 8),  "rainfall": 85.0, "soil": "Saturated", "water_visible": True},
    {"district": "Balochistan_District", "date": date(2022, 9, 5),  "rainfall": 60.0, "soil": "Saturated", "water_visible": True},
    {"district": "KP_District",          "date": date(2024, 6, 29), "rainfall": 70.0, "soil": "Saturated", "water_visible": True},
    {"district": "Nowshera",             "date": date(2024, 7, 15), "rainfall": 60.0, "soil": "Saturated", "water_visible": True},
    {"district": "Buner",                "date": date(2024, 8, 15), "rainfall": 75.0, "soil": "Saturated", "water_visible": True},
    {"district": "Hyderabad",            "date": date(2022, 8, 30), "rainfall": 55.0, "soil": "Saturated", "water_visible": True},
    {"district": "Karachi",              "date": date(2022, 9, 1),  "rainfall": 45.0, "soil": "Moist",     "water_visible": True},
    {"district": "Lahore",               "date": date(2024, 7, 1),  "rainfall": 12.0, "soil": "Dry",       "water_visible": False},
]

@st.cache_data(show_spinner=False)
def run_alert_scan():
    """Scan all monitored districts and return only those at High/Critical risk."""
    active = []
    for m in LIVE_MONITORING:
        try:
            r = predict_flood_risk(
                date=str(m["date"]),
                district=m["district"],
                rainfall_today=m["rainfall"],
                soil_condition=m["soil"],
                water_visible=m["water_visible"],
            )
            if r["risk_level"] in ("High", "Critical"):
                active.append({
                    "district":   m["district"],
                    "risk_level": r["risk_level"],
                    "confidence": r["confidence"],
                })
        except Exception:
            continue
    # Sort Critical first, then High
    active.sort(key=lambda x: 0 if x["risk_level"] == "Critical" else 1)
    return active

# Header for the alert section
st.markdown("### 🚨 Active Flood Alerts — System Overview")
alert_col1, alert_col2 = st.columns([5, 1], gap="medium")
with alert_col1:
    st.caption(f"Scanning {len(LIVE_MONITORING)} monitored districts. Districts at High or Critical risk are listed below with recommended action.")
with alert_col2:
    rescan = st.button("🔄 Re-scan", use_container_width=True, key="rescan_btn")

if rescan:
    st.cache_data.clear()  # force refresh

active_alerts = run_alert_scan()

if active_alerts:
    # Summary line
    crit_count = sum(1 for a in active_alerts if a["risk_level"] == "Critical")
    high_count = sum(1 for a in active_alerts if a["risk_level"] == "High")
    summary_parts = []
    if crit_count: summary_parts.append(f"**{crit_count} CRITICAL**")
    if high_count: summary_parts.append(f"**{high_count} HIGH**")
    st.markdown(f"#### {' · '.join(summary_parts)} risk districts require immediate action")

    # Render each alert as a compact card
    for alert in active_alerts:
        render_compact_alert_card(alert)
else:
    st.success("✓ No districts currently at High or Critical risk. System monitoring active.")

st.divider()


# -----------------------------------------------------------------------------
# DEMO SCENARIO ROW
# -----------------------------------------------------------------------------
st.markdown("##### 📋 Quick demo scenarios")
demo_cols = st.columns(len(DEMO_SCENARIOS))
for i, (name, scenario) in enumerate(DEMO_SCENARIOS.items()):
    if demo_cols[i].button(name, use_container_width=True, key=f"demo_{i}"):
        apply_scenario(scenario)
        st.rerun()

st.divider()


# -----------------------------------------------------------------------------
# TWO-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col_input, col_output = st.columns([1, 2], gap="large")


# =============================================================================
# INPUT PANEL
# =============================================================================
with col_input:
    st.markdown("### Inputs")

    input_date = st.date_input(
        "Date", key="in_date",
        min_value=date(2022, 1, 1), max_value=date(2026, 12, 31),
        help="Date for which to assess flood risk",
    )
    district = st.selectbox(
        "District", options=DISTRICT_LIST, key="in_district",
        format_func=lambda d: DISTRICT_DISPLAY.get(d, d),
        help="Districts marked 'training data' have direct model coverage; "
             "others use regional analogues with reduced confidence.",
    )
    rainfall = st.number_input(
        "Rainfall today (mm)", key="in_rainfall",
        min_value=0.0, max_value=5000.0, step=1.0,
        help="Total rainfall recorded today in millimeters",
    )
    soil_condition = st.radio(
        "Soil condition", options=["Dry", "Moist", "Saturated"], key="in_soil",
        horizontal=True, help="Current ground saturation level",
    )
    water_visible = st.checkbox(
        "Visible standing water reported", key="in_water",
        help="Local report of standing water on the ground",
    )

    st.markdown("&nbsp;")
    submit = st.button("Assess Flood Risk", type="primary", use_container_width=True)


# =============================================================================
# OUTPUT PANEL
# =============================================================================
with col_output:
    if not submit:
        st.markdown("### Prediction")
        st.info("👈 Enter inputs on the left and click **Assess Flood Risk** to see prediction. "
                "Or click a demo scenario above to load preset inputs.")

    else:
        st.markdown("### Prediction")

        with st.spinner("Computing prediction..."):
            try:
                result = predict_flood_risk(
                    date=str(input_date),
                    district=district,
                    rainfall_today=rainfall,
                    soil_condition=soil_condition,
                    water_visible=water_visible,
                )
            except ValueError as e:
                st.error(f"Could not compute prediction: {e}")
                st.stop()

        # Render in this order:
        render_alert_mode_banner(result, district)   # ← Scenario 3: only shows for High/Critical
        render_risk_banner(result)             # ← banner up top
        render_district_context_card(result)   # ← Phase D-2: 4-metric context grid
        render_action_card(result)
        render_drivers_card(result)
        render_probs_card(result)
        render_warning(result)

        # ── Charts ─────────────────────────────────────────────────────────
        st.markdown("&nbsp;")
        chart_col_left, chart_col_right = st.columns([1, 1], gap="medium")
        with chart_col_left:
            st.markdown('<div class="output-card-title">🗺️ District Map</div>',
                        unsafe_allow_html=True)
            display_label = DISTRICT_DISPLAY.get(district, district).split("  ")[0]
            map_fig = fc.build_pakistan_map(
                selected_district=district,
                risk_level=result["risk_level"],
                display_name=display_label,
            )
            st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

        with chart_col_right:
            st.markdown('<div class="output-card-title">📈 Recent Trend</div>',
                        unsafe_allow_html=True)
            timeline_fig = fc.build_30day_timeline(
                effective_district=result["effective_district"],
                end_date=input_date,
                lookup_df=LOOKUP_DF,
                oof_preds_onset=OOF_PREDS_ONSET,
            )
            st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="output-card-title">🎯 Feature Contributions</div>',
                    unsafe_allow_html=True)
        shap_fig = fc.build_shap_chart(result["explanation"])
        st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})