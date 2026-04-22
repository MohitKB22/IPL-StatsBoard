#!/usr/bin/env bash
# =============================================================================
#  IPL Dashboard — Quick Start Script
# =============================================================================
set -e

echo ""
echo "======================================================"
echo "  🏏  IPL Advanced Analytics Dashboard"
echo "======================================================"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PY=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
MAJOR=$(echo "$PY" | cut -d. -f1)
MINOR=$(echo "$PY" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
  echo "❌  Python 3.9+ required (found $PY)"
  exit 1
fi
echo "✅  Python $PY detected"

# ── Virtual environment (optional) ───────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo ""
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅  Virtual environment active"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅  Dependencies installed"

# ── Data check ───────────────────────────────────────────────────────────────
echo ""
if [ ! -f "data/matches.csv" ] || [ ! -f "data/deliveries.csv" ]; then
  echo "⚠️   Data files not found in data/"
  echo ""
  echo "  Option A — Use synthetic demo data (generated now):"
  echo "    python generate_sample_data.py"
  echo ""
  echo "  Option B — Use real IPL data from Kaggle:"
  echo "    https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020"
  echo "    Place matches.csv + deliveries.csv in data/"
  echo ""
  read -p "Generate synthetic demo data now? [Y/n] " choice
  choice=${choice:-Y}
  if [[ "$choice" =~ ^[Yy]$ ]]; then
    echo ""
    python generate_sample_data.py
  else
    echo "Skipping data generation. Add your CSVs to data/ and re-run."
    exit 0
  fi
else
  echo "✅  Data files found"
fi

# ── Run tests (optional) ──────────────────────────────────────────────────────
echo ""
read -p "Run test suite before launching? [y/N] " run_tests
run_tests=${run_tests:-N}
if [[ "$run_tests" =~ ^[Yy]$ ]]; then
  echo ""
  python -m pytest tests/ -v --tb=short
fi

# ── Launch app ────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  🚀  Launching Streamlit dashboard..."
echo "      http://localhost:8501"
echo "======================================================"
echo ""
streamlit run app.py
