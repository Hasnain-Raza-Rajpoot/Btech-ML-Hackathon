"""
FloodSense charts (Phase C, fixes applied)
============================================
Three Plotly charts that get rendered alongside the prediction output:
  1. Pakistan map with the selected district highlighted in the risk color
  2. 30-day rainfall + flood probability timeline for the selected district
  3. SHAP horizontal bar chart of top drivers
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

HERE       = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
MODELS_DIR = os.path.join(HERE, "final_models")


# -----------------------------------------------------------------------------
# DISTRICT GEOGRAPHIC COORDINATES
# -----------------------------------------------------------------------------
DISTRICT_COORDS = {
    "Sindh_District":        (25.50, 68.00),
    "Balochistan_District":  (28.50, 65.00),
    "KP_District":           (34.00, 72.00),
    # Sindh
    "Karachi": (24.86, 67.01), "Hyderabad": (25.39, 68.37),
    "Larkana": (27.55, 68.21), "Sukkur": (27.71, 68.85),
    "Mirpur Khas": (25.53, 69.01), "Thatta": (24.74, 67.92),
    "Dadu": (26.73, 67.78), "Sanghar": (26.03, 68.95),
    # Punjab
    "Lahore": (31.55, 74.34), "Faisalabad": (31.42, 73.08),
    "Multan": (30.20, 71.47), "Rawalpindi": (33.60, 73.05),
    "Bahawalpur": (29.40, 71.69), "Sialkot": (32.49, 74.53),
    "Gujranwala": (32.16, 74.18), "Dera Ghazi Khan": (30.05, 70.64),
    # KP
    "Peshawar": (34.01, 71.58), "Nowshera": (34.02, 71.97),
    "Buner": (34.42, 72.65), "Swat": (34.78, 72.36),
    "Charsadda": (34.15, 71.74), "Mansehra": (34.33, 73.20),
    "Mardan": (34.20, 72.04), "Dera Ismail Khan": (31.82, 70.90),
    "Abbottabad": (34.16, 73.22), "Kohat": (33.59, 71.44),
    # Balochistan
    "Quetta": (30.18, 66.99), "Lasbela": (26.21, 65.66),
    "Khuzdar": (27.81, 66.62), "Sibi": (29.55, 67.88),
    "Turbat": (26.00, 63.05), "Gwadar": (25.13, 62.33),
    "Kech": (26.00, 62.50),
    # GB / AJK
    "Gilgit": (35.92, 74.31), "Skardu": (35.30, 75.63),
    "Muzaffarabad": (34.36, 73.47),
}


RISK_COLORS = {
    "Low":      "#16A34A",
    "Medium":   "#EAB308",
    "High":     "#EA580C",
    "Critical": "#DC2626",
}

# Standard chart height across map and timeline
CHART_HEIGHT = 400


# -----------------------------------------------------------------------------
# 1. PAKISTAN MAP
# -----------------------------------------------------------------------------
def build_pakistan_map(selected_district, risk_level, display_name=None):
    other_lats, other_lons, other_names = [], [], []
    for name, (lat, lon) in DISTRICT_COORDS.items():
        if name == selected_district:
            continue
        other_lats.append(lat)
        other_lons.append(lon)
        other_names.append(name)

    sel_lat, sel_lon = DISTRICT_COORDS.get(selected_district, (30.0, 70.0))
    risk_color = RISK_COLORS.get(risk_level, "#6B7280")
    label = display_name or selected_district

    fig = go.Figure()

    # Other districts — small grey dots
    fig.add_trace(go.Scattermapbox(
        lat=other_lats, lon=other_lons, mode="markers",
        marker=dict(size=8, color="#9CA3AF", opacity=0.6),
        text=other_names, hoverinfo="text",
        showlegend=False,
    ))

    # Halo around selected district
    fig.add_trace(go.Scattermapbox(
        lat=[sel_lat], lon=[sel_lon], mode="markers",
        marker=dict(size=44, color=risk_color, opacity=0.25),
        hoverinfo="skip", showlegend=False,
    ))

    # Selected district marker + label
    fig.add_trace(go.Scattermapbox(
        lat=[sel_lat], lon=[sel_lon], mode="markers+text",
        marker=dict(size=22, color=risk_color, opacity=0.95),
        text=[label], textposition="top center",
        textfont=dict(size=14, color="#1F2937", family="Inter, sans-serif"),
        hovertext=[f"{label}<br>Risk: {risk_level}"],
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=30.5, lon=70.5),
            zoom=4.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=CHART_HEIGHT,
        showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# 2. 30-DAY TIMELINE — fixed: no in-chart title (header is in app), legend below,
#    bigger margins for axis labels, "today" line, flood-day highlights
# -----------------------------------------------------------------------------
def build_30day_timeline(effective_district, end_date, lookup_df, oof_preds_onset):
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=30)

    sub = lookup_df[(lookup_df["district"] == effective_district) &
                     (lookup_df["date"] >= start_date) &
                     (lookup_df["date"] <= end_date)].sort_values("date")
    sub = sub.drop_duplicates(subset="date", keep="first")

    if len(sub) == 0:
        sub = (lookup_df[lookup_df["district"] == effective_district]
               .sort_values("date").tail(30).drop_duplicates(subset="date", keep="first"))

    if oof_preds_onset is not None and len(sub) > 0:
        sub_idx = sub.index.values
        onset_probs = oof_preds_onset[sub_idx]
        onset_probs = np.where(np.isnan(onset_probs), 0, onset_probs)
    else:
        onset_probs = np.zeros(len(sub))

    fig = go.Figure()

    # Highlight flood days as background bands (light red)
    if "flood_event" in sub.columns:
        for _, row in sub[sub["flood_event"] == 1].iterrows():
            d = row["date"]
            fig.add_vrect(
                x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12),
                fillcolor="#FEE2E2", opacity=0.5, line_width=0, layer="below",
            )

    # Rainfall bars (left axis)
    fig.add_trace(go.Bar(
        x=sub["date"], y=sub["precipitation"],
        name="Rainfall (mm)",
        marker=dict(color="#3B82F6", opacity=0.7),
        hovertemplate="<b>%{x|%b %d}</b><br>Rainfall: %{y:.1f} mm<extra></extra>",
        yaxis="y",
    ))

    # Onset probability line (right axis)
    fig.add_trace(go.Scatter(
        x=sub["date"], y=onset_probs,
        name="Onset probability",
        mode="lines+markers",
        line=dict(color="#DC2626", width=2.5),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d}</b><br>Onset prob: %{y:.2%}<extra></extra>",
        yaxis="y2",
    ))

    # Threshold lines on probability axis (left-side annotation to avoid clipping)
    fig.add_hline(
        y=0.3, line_dash="dot", line_color="#F59E0B", opacity=0.7,
        annotation_text="Medium (0.30)",
        annotation_position="top left",
        annotation_font_size=9,
        yref="y2",
    )
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="#DC2626", opacity=0.7,
        annotation_text="High (0.50)",
        annotation_position="top left",
        annotation_font_size=9,
        yref="y2",
    )

    # "Today" line — vertical marker at end_date
    # NOTE: using add_shape + add_annotation instead of add_vline because
    # add_vline crashes when given pandas Timestamps + annotation_text together
    # (Plotly tries to compute mean of [timestamp] which fails on modern pandas)
    fig.add_shape(
        type="line",
        x0=end_date, x1=end_date,
        y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="#1F2937", width=1),
        opacity=0.5,
    )
    fig.add_annotation(
        x=end_date, y=1.0, yref="paper",
        text="Today", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(size=10, color="#1F2937"),
        xshift=4,
    )

    fig.update_layout(
        # Title removed — section header above the chart already says "RECENT TREND"
        xaxis=dict(
            showgrid=False,
            tickformat="%b %d",
        ),
        yaxis=dict(
            title=dict(text="Rainfall (mm)", font=dict(color="#3B82F6", size=12)),
            tickfont=dict(color="#3B82F6", size=10),
            showgrid=True, gridcolor="#F3F4F6",
        ),
        yaxis2=dict(
            title=dict(text="Onset probability", font=dict(color="#DC2626", size=12)),
            tickfont=dict(color="#DC2626", size=10),
            overlaying="y", side="right",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.30,    # below chart
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="white",
        margin=dict(l=70, r=70, t=20, b=70),  # roomier left/right + bottom for legend
        height=CHART_HEIGHT,
        hovermode="x unified",
    )
    return fig


# -----------------------------------------------------------------------------
# 3. SHAP HORIZONTAL BAR CHART
# -----------------------------------------------------------------------------
def build_shap_chart(drivers):
    drivers = list(reversed(drivers))

    labels = [d["label"] for d in drivers]
    shap_values = [d["shap"] for d in drivers]
    raw_values = [d["value"] for d in drivers]

    colors = ["#DC2626" if s > 0 else "#16A34A" for s in shap_values]

    text_labels = [
        f"{lab}<br>(value = {v:.2f})" for lab, v in zip(labels, raw_values)
    ]

    fig = go.Figure(go.Bar(
        x=shap_values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{s:+.2f}" for s in shap_values],
        textposition="outside",
        textfont=dict(size=12, color="#1F2937"),
        hovertext=text_labels, hoverinfo="text",
        showlegend=False,
    ))

    fig.add_vline(x=0, line_color="#1F2937", line_width=1)

    fig.update_layout(
        # Title removed — section header above chart already says "FEATURE CONTRIBUTIONS"
        xaxis=dict(
            title="SHAP value (positive = pushes risk UP, negative = pushes DOWN)",
            zeroline=True, zerolinecolor="#1F2937", zerolinewidth=1,
            gridcolor="#F3F4F6",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color="#1F2937"),
            automargin=True,
        ),
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=20, b=50),
        height=320,
    )
    return fig