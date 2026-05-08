# 🌊 FloodSense

**AI-powered flood early-warning system for Pakistan**
*Built for the Neural Nova 60-Hour Data Drop hackathon (BTech).*

🔗 **Live demo:** [btech-ml-hackathon-floodsense-app.streamlit.app](https://btech-ml-hackathon-floodsense-app.streamlit.app)

---

## Why this matters

The **2022 Pakistan floods displaced 33 million people** — roughly 1 in 7 Pakistanis. NDMA reports show that the difference between a tragedy and a disaster often came down to hours of advance warning, not days. District officials in places like Nowshera, Sindh, and Balochistan needed an *actionable* signal: not just "flood risk is elevated," but "based on what's happening today, evacuate these zones now."

FloodSense is built for that decision. Five inputs (date, district, rainfall, soil saturation, visible standing water) → one risk band (Low / Medium / High / Critical) → a recommended action and a transparent explanation of *why* the model decided what it did.

---

## What the app delivers

| Component | What it shows |
|---|---|
| **Risk banner** | Risk level (color-coded) + confidence score |
| **District context** | Population, estimated population at risk, last major flood, 7-day risk trajectory |
| **Recommended action** | Operational guidance scaled to risk level |
| **Why this prediction?** | Top 5 SHAP drivers in plain English (e.g., "Water extent 2 weeks ago: 963.41") |
| **Pakistan map** | Selected district highlighted in risk color, all 39 districts visible |
| **30-day timeline** | Rainfall bars + onset probability line, with flood-history bands |
| **SHAP bar chart** | Quantified contribution of each top driver |

The app handles **39 Pakistani districts** across all 4 provinces + Gilgit-Baltistan + AJK. For districts outside the model's direct training data (36 of the 39), predictions use a regional analogue with a **30% confidence haircut** and a clear warning banner — the model is honest about where it's extrapolating.

---

## Headline results

| Metric | Score |
|---|---|
| Cross-district AUC (continuation model) | **0.9689** |
| Cross-district AUC (onset model) | **0.9756** |
| Same-day onset detection (Medium+ threshold) | **89%** of 134 historical events |
| Same-day onset detection (High+ threshold) | **78%** of events |
| Cross-validation method | GroupKFold by district (3 folds) |
| Brief's accuracy threshold | 70% — comfortably exceeded |

Validation was strict: **every district was held out for testing**, so AUC reflects genuine cross-district generalization, not memorization of training districts.

---

## The methodology story

What makes FloodSense more than a model: we diagnosed and fixed eight serious data issues before training. The README of this repo is also a methodological log — judges valuing rigor will find it documented.

### Eight landmines we found and fixed

1. **`ds_idx` leak** — a column that uniquely identified each row, would have given perfect train AUC and zero generalization
2. **Date format inconsistency** — MM/DD/YYYY vs DD/MM/YYYY mixed in raw data
3. **Phantom rows** — duplicate dates with conflicting values
4. **Year-district confound** — *each of the 3 districts appears in only ONE year* (Sindh: 2022, Balochistan: 2023, KP: 2024). A naive chronological split would test the model on a totally different district from what it trained on.
5. **`water_area_km2` perfect predictor** — flagged the leak: AUC 1.0 per fold using this as a feature. Removed; baseline AUC dropped to 0.55.
6. **Provided rolling averages broken** — 29% disagreement with our recomputed values; we recomputed all rolling features ourselves.
7. **Data spans not aligned across districts** — handled with forward/backward fallback in production.
8. **Class imbalance distorted by year-district confound** — solved by switching CV strategy.

### Two models, not one

After feature engineering pushed cross-district AUC to 0.96, **a sensitivity test revealed an uncomfortable truth**: removing all `water_area` lag features dropped AUC to 0.58. The model was effectively a *flood-continuation detector* — it knew floods were ongoing because two-week-old flood water was still visible. That's useful (most flood deaths happen during continuation, not onset), but it's not what officials need for evacuation timing.

So we built a **second model: an onset detector**. Trained only on (non-flood + day-of-onset) cases, with a new target. AUC: 0.9573 cross-district, **with onset probability cleanly distinguishing onset events (0.64) from continuation days (0.07) and non-flood days (0.05)**.

The deployed app uses both models. A prediction is labeled `new_onset`, `ongoing`, `rising_risk`, or `no_flood` based on which probabilities fire above threshold.

### Honesty about what FloodSense is and isn't

- ✅ **Same-day detection**: 89% Medium+ recall on 134 historical onset events
- ⚠️ **Multi-day forecasting**: only 26% of events flagged Medium 1-3 days in advance
- ⚠️ **Model leans heavily on `water_area_km2_lag14`** — the most influential SHAP feature is "water extent two weeks ago," meaning the model needs satellite-observed water history to function
- ⚠️ **Punjab, GB, AJK are extrapolations** — only 3 districts in training data, all 36 others use regional analogues

These limitations are surfaced in the UI (warning banners) and in the recommended action (advice scales to risk level, never overpromises).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  5 inputs     │  │  Prediction      │  │  3 Plotly       │  │
│  │  + 4 demos    │→ │  + risk banner   │+ │  charts (map,   │  │
│  │               │  │  + context card  │  │  timeline,      │  │
│  │               │  │  + drivers       │  │  SHAP)          │  │
│  └───────────────┘  └──────────────────┘  └─────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ floodsense_predict.py   │
                    │  • feature lookup       │
                    │  • user signal cascade  │
                    │  • two-model inference  │
                    │  • SHAP explanation     │
                    │  • district context     │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   ┌────▼──────┐         ┌───────▼──────┐         ┌────────▼─────┐
   │ LightGBM  │         │ LightGBM     │         │ floodsense_  │
   │ continua- │         │ onset        │         │ onset.csv    │
   │ tion (.txt│         │ (.txt)       │         │ (1365 rows × │
   │ portable) │         │              │         │  ~50 features)│
   └───────────┘         └──────────────┘         └──────────────┘
        │                        │
        └─────────┬──────────────┘
                  │
            ┌─────▼──────┐
            │ SHAP Tree  │
            │ Explainer  │  (rebuilt at runtime — no pickle deps)
            │  (47 feat) │
            └────────────┘
```

**Why this design:**

- **LightGBM `.txt` model files** are language-agnostic and version-stable. The deployed app rebuilds SHAP explainers from these models at startup — no pickle compatibility issues across environments.
- **Two models running in parallel** so the UI can show *both* onset and continuation probabilities, letting officials distinguish "new flood is starting" from "ongoing flood is sustained."
- **User input cascade** — when a user reports "visible standing water" + "saturated soil" + "30+ mm rainfall," the cascade overrides relevant lag features with district flood-day medians. This is what makes the UI inputs actually move the prediction (otherwise lag features would dominate and override what the user sees).

---

## Run locally

```bash
# Clone the repo
git clone https://github.com/Hasnain-Raza-Rajpoot/Btech-ML-Hackathon.git
cd Btech-ML-Hackathon/floodsense_app

# Install dependencies (Python 3.10 recommended)
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

App opens at `http://localhost:8501`. Click any demo button to see a flood scenario.

### Reproduce the ML pipeline

The full training pipeline is in numbered Jupyter notebooks at the repo root:

```
1. data_cleaning.ipynb      — fix the 8 landmines
2. EDA.ipynb                — exploratory analysis, year-district confound
3. baseline_model.ipynb     — confirm leakage, establish floor
4. feature_engineering.ipynb — 47 features, AUC 0.9560
5. sensitivity_test.ipynb   — discover continuation-vs-onset distinction
6. onset_model.ipynb        — second model
7. historical_backtest.ipynb — 134 events, 89% same-day recall
8. hyperparameter_tuning.ipynb — Optuna 40 trials → 0.9689 / 0.9756
9. shap_analysis.ipynb      — interpretability
```

---

## Repository structure

```
Btech-ML-Hackathon/
│
├── floodsense_app/          # Deployed Streamlit app (Phase F)
│   ├── app.py               # UI (570 lines)
│   ├── floodsense_predict.py    # Inference + SHAP + context (~440 lines)
│   ├── floodsense_charts.py     # Plotly charts (~240 lines)
│   ├── floodsense_onset.csv     # Engineered features lookup (~950 KB)
│   ├── final_models/        # Production model artifacts
│   ├── requirements.txt
│   ├── .streamlit/config.toml   # Pakistani green theme
│   └── README.md
│
├── *.ipynb                  # 9 development notebooks
│
├── final_models/            # Trained model artifacts (master copy)
│   ├── lgb_continuation_final.txt   # LightGBM portable model file
│   ├── lgb_onset_final.txt
│   ├── feature_cols.pkl
│   ├── oof_preds_*.npy      # Out-of-fold predictions for trajectory
│   └── shap_*.csv           # Feature importance rankings
│
├── processed/               # Cleaned + engineered datasets
├── eda_plots/, baseline_plots/, ...   # Visualization outputs
├── Instructions/            # Original hackathon brief + data dictionary
└── uploads/                 # Original raw data
```

---

## Tech stack

- **ML**: LightGBM, SHAP, scikit-learn, Optuna
- **Data**: pandas, NumPy
- **UI**: Streamlit, Plotly, custom CSS
- **Deploy**: Streamlit Community Cloud (Python 3.10)

---

## Reading order for judges

If you're reviewing this submission and have limited time:

1. **Click the live demo first** → [btech-ml-hackathon-floodsense-app.streamlit.app](https://btech-ml-hackathon-floodsense-app.streamlit.app) → click any demo button
2. **Read this README** (you're here)
3. **Skim `landmines.ipynb`** — the 8-issue data audit, with quantified impact of each fix
4. **Skim `sensitivity_test.ipynb`** — the moment we realized the model was a continuation detector and what we did about it
5. **Skim `historical_backtest.ipynb`** — 134-event recall analysis with case studies for Sindh 2022, Balochistan 2023, and KP 2024 floods

---

## Acknowledgements

Built solo for the Neural Nova 60-Hour Data Drop sprint by BTech.

Data sources:
- Hackathon-provided flood sensor data (1,434 daily records, 2022–2024)
- NDMA Pakistan regional impact data (2022 floods)
- NASA SRTM elevation data

Population estimates for "Estimated at risk" metric are census-era approximations and serve as planning indicators, not precise counts.

---

*Built for a Better Tomorrow — Neural Nova × BTech*