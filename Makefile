# =============================================================================
#  IPL Dashboard — Makefile
#  Usage: make <target>
# =============================================================================

.PHONY: help install data test lint format run clean reset

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
STREAMLIT:= $(VENV)/bin/streamlit
RUFF     := $(VENV)/bin/ruff

# ── Default ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🏏  IPL Dashboard — Available targets"
	@echo ""
	@echo "  make install    Install all dependencies in a virtualenv"
	@echo "  make data       Generate synthetic sample data"
	@echo "  make test       Run the full test suite"
	@echo "  make lint       Lint with ruff"
	@echo "  make format     Auto-format with ruff"
	@echo "  make run        Launch the Streamlit app"
	@echo "  make train      Pre-train & cache ML models"
	@echo "  make clean      Remove __pycache__ and .pytest_cache"
	@echo "  make reset      Remove data + cached models (fresh start)"
	@echo ""

# ── Environment ────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	$(PIP) install ruff -q
	@echo "✅  Dependencies installed"

# ── Data ───────────────────────────────────────────────────────────────────────
data:
	$(PYTHON) generate_sample_data.py

# ── Testing ────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -x -q

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ app.py generate_sample_data.py

format:
	$(RUFF) format src/ app.py generate_sample_data.py

# ── App ────────────────────────────────────────────────────────────────────────
run:
	$(STREAMLIT) run app.py

# ── Pre-train models ──────────────────────────────────────────────────────────
train:
	$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
import warnings; warnings.filterwarnings('ignore'); \
from src.data_loader import load_all_data; \
from src.preprocessing import get_clean_data; \
from src.model import MatchWinnerModel; \
from src.win_probability import WinProbabilityModel; \
m_raw, d_raw = load_all_data(); \
m, d, merged = get_clean_data(m_raw, d_raw); \
print('Training MatchWinnerModel...'); \
mw = MatchWinnerModel(); acc = mw.train(m); mw.save(); print(f'  Accuracy: {acc}%'); \
print('Training WinProbabilityModel...'); \
wp = WinProbabilityModel(); auc = wp.train(merged); wp.save(); print(f'  AUC: {auc}'); \
print('Models cached to data/*.pkl ✅'); \
"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅  Cleaned"

reset: clean
	rm -f data/matches.csv data/deliveries.csv
	rm -f data/*.pkl
	@echo "✅  Reset to fresh state (run 'make data' to regenerate)"
