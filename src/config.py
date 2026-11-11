"""Project configuration: experiment design parameters and simulation "ground truth".

All paths are relative to the repository root, so the project runs on any machine.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
RESULTS_DIR = OUTPUTS_DIR / "results"

RAW_DATA_PATH = DATA_DIR / "raw_experiment_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_experiment_data.csv"

CONFIG = {
    "project": "Personalized Recommendations A/B Test",
    "project_ar": "اختبار A/B لنموذج التوصيات المخصصة",

    # ---------------- Design (known BEFORE the experiment) ----------------
    "alpha": 0.05,                    # significance level (two-sided)
    "power": 0.80,                    # target statistical power
    "baseline_session_min": 20.0,     # current average session duration (minutes), from historical data
    "baseline_session_sd": 16.0,      # historical SD of user-level average session duration
    "mde_rel": 0.05,                  # smallest relative lift that changes the decision (+5% = +1 minute)
    "hypothesized_lift_rel": 0.15,    # the lift stated in the hypothesis (+15%)
    "daily_new_users": 1_000,         # users entering the experiment per day
    "treatment_share": 0.5,           # 50% control (A) / 50% treatment (B)
    "min_weeks": 2,                   # run at least two full weekly cycles
    "assignment_salt": "reco-v2-2026",  # salt for deterministic hash-based assignment
    "srm_threshold": 0.001,           # p-value below which we declare Sample Ratio Mismatch

    # Guardrail non-inferiority margins (worst acceptable degradation for B vs A)
    "churn_margin_abs": 0.005,        # churn may not rise by more than +0.5 percentage points
    "latency_margin_ms": 25.0,        # mean server latency may not rise by more than +25 ms

    # Business translation
    "annual_users": 600_000,          # users per year affected by the decision

    # ---------------- Simulation "truth" (unknown to the analyst in reality) ----------------
    "seed": 2026,
    "true_lift_rel": 0.08,            # true average lift in session duration (+8%)
    "novelty_boost": 0.5,             # extra relative lift in the first days, decaying (novelty effect)
    "novelty_decay_days": 3.0,
    "mobile_share": 0.65,
    "mobile_lift_multiplier": 1.10,   # heterogeneity: effect slightly larger on mobile
    "web_lift_multiplier": 0.80,
    "base_ctr": 0.060,                # click-through rate on recommendation widgets
    "ctr_effect_rel": 0.20,           # +20% relative CTR in B
    "items_per_session": 4.0,
    "items_effect_rel": 0.05,
    "churn_rate": 0.040,              # 14-day subscription cancellation rate
    "churn_effect_abs": -0.002,       # B slightly reduces churn
    "latency_mean_ms": 180.0,
    "latency_sd_ms": 40.0,
    "latency_effect_ms": 12.0,        # the new model adds ~12 ms server latency
    "duplicate_share": 0.003,         # logging bug: 0.3% of rows duplicated
    "missing_latency_share": 0.01,    # 1% of users have no latency measurement
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHARTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
