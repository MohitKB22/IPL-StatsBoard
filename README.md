# 🏏 IPL Advanced Analytics Dashboard (2008–2025)

A professional-grade interactive sports-analytics platform built with Python,
Streamlit, Plotly, and Scikit-learn.

---

## ✨ Live Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI strip, feature cards, latest season snapshot |
| 👤 **Players** | Leaderboards · career arcs · phase breakdown · Best XI · head-to-head |
| 🏆 **Teams** | Win % · H2H records · season timeline · toss impact per team |
| 📈 **Trends** | Season evolution · over breakdown · phase analysis · score worm · boundary heatmap |
| 🏟️ **Venues** | Scoring profiles · pitch tendencies · advantage score · team-venue heatmap |
| 📊 **Win Probability** | Ball-by-ball GradientBoosting · momentum chart · live simulator · event markers |
| 🤖 **Prediction** | Pre-match RandomForest · gauge output · what-if explorer · feature importance |

---

## 📂 Project Structure

```
ipl-dashboard/
├── app.py                        ← Streamlit home / landing page
├── pages/                        ← Multi-page Streamlit pages
│   ├── _shared.py                ← Shared CSS, data loader, sidebar helpers
│   ├── 1_👤_Players.py
│   ├── 2_🏆_Teams.py
│   ├── 3_🏟️_Venues.py
│   ├── 4_📈_Trends.py
│   ├── 4_📊_Win_Probability.py
│   └── 5_🤖_Prediction.py
├── src/                          ← Backend modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── model.py                  ← RandomForest match predictor
│   └── win_probability.py        ← GradientBoosting win probability
├── tests/
│   └── test_pipeline.py          ← 39-test suite
├── notebooks/
│   └── eda.py                    ← Exploratory analysis
├── data/                         ← CSV + cached .pkl models
├── .streamlit/config.toml        ← Dark gold theme
├── .github/workflows/ci.yml      ← GitHub Actions CI
├── generate_sample_data.py       ← Synthetic data generator
├── pyproject.toml
├── Makefile
├── setup.sh
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# One command
bash setup.sh

# Or manually
pip install -r requirements.txt
python generate_sample_data.py   # or add real Kaggle CSVs to data/
streamlit run app.py
```

## 🤖 ML Models

| Model | Algorithm | Target | Metric |
|-------|-----------|--------|--------|
| Match Winner | RandomForest (200 trees) | team1 wins? | ~55-65% acc |
| Win Probability | GradientBoosting (300 trees) | batting team wins? | AUC ~0.87-0.92 |

## 🧪 Tests

```bash
python -m pytest tests/ -v   # 39 tests, all passing
```

## ☁️ Deploy

Push to GitHub → connect to [share.streamlit.io](https://share.streamlit.io) → set `app.py` as entry point.

---
*MIT License · Built with Streamlit, Plotly, Scikit-learn*
