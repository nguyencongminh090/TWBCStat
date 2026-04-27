"""
Output path constants — single source of truth.
All modules import from here so folder structure changes
only need to be made in one place.

Output structure:
  output/
  ├── data/           ← all .csv files
  ├── plots/
  │   ├── ratings/    ← elo, trueskill, BT, ensemble plots
  │   ├── model/      ← calibration, ROC, regression, heatmaps
  │   ├── monte_carlo/← match simulations
  │   └── trends/     ← per-player win trend charts
  └── reports/        ← .txt reports
"""
import os

_BASE = os.path.join(os.path.dirname(__file__), "..")

# Root output directory
OUT        = os.path.join(_BASE, "output")

# Sub-directories
DATA_DIR   = os.path.join(OUT, "data")
PLOTS_DIR  = os.path.join(OUT, "plots")
RATINGS_DIR  = os.path.join(PLOTS_DIR, "ratings")
MODEL_DIR    = os.path.join(PLOTS_DIR, "model")
MC_DIR       = os.path.join(PLOTS_DIR, "monte_carlo")
TRENDS_DIR   = os.path.join(PLOTS_DIR, "trends")
REPORTS_DIR  = os.path.join(OUT, "reports")


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, RATINGS_DIR, MODEL_DIR, MC_DIR, TRENDS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Convenience helpers ──────────────────────────────────────
def data(filename):   return os.path.join(DATA_DIR,    filename)
def plot_r(filename): return os.path.join(RATINGS_DIR, filename)
def plot_m(filename): return os.path.join(MODEL_DIR,   filename)
def plot_mc(filename):return os.path.join(MC_DIR,      filename)
def report(filename): return os.path.join(REPORTS_DIR, filename)
