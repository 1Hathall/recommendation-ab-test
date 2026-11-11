"""Step 3 - Data preparation and trust checks.

Checks: schema & completeness, duplicate records, missing values, value ranges,
assignment integrity (re-computing the hash), Sample Ratio Mismatch (SRM, chi-square),
and covariate balance on pre-experiment variables (standardized mean difference).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import CLEAN_DATA_PATH, CONFIG, RAW_DATA_PATH, RESULTS_DIR
from .simulate_data import assign_variant

REQUIRED_COLUMNS = [
    "user_id", "variant", "enroll_day", "device", "pre_avg_session_min_30d", "avg_session_min",
    "sessions_7d", "items_viewed_per_session", "rec_impressions", "rec_clicks", "churned_14d", "avg_latency_ms",
]


def srm_check(n_a: int, n_b: int, expected_share_b: float = 0.5) -> dict:
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], f_exp=[total * (1 - expected_share_b), total * expected_share_b])
    return {"n_A": int(n_a), "n_B": int(n_b), "share_B": n_b / total, "chi2": float(chi2), "p_value": float(p),
            "srm_detected": bool(p < CONFIG["srm_threshold"])}


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference; |SMD| < 0.1 is the usual balance threshold."""
    return float((b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def run() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(RAW_DATA_PATH)
    report: dict = {"raw_rows": int(len(raw))}

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 1) Duplicates: exact duplicate rows (logging bug) and any user_id appearing twice
    exact_dups = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    conflicting_ids = int(df["user_id"].duplicated().sum())
    df = df.drop_duplicates(subset="user_id", keep="first")
    report.update({"exact_duplicate_rows_removed": exact_dups, "conflicting_user_ids_removed": conflicting_ids})

    # 2) Missing values (latency is a guardrail only -> excluded from that metric, not imputed)
    report["missing_values"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    # 3) Range / logic checks
    checks = {
        "negative_session_time": int((df["avg_session_min"] < 0).sum()),
        "clicks_gt_impressions": int((df["rec_clicks"] > df["rec_impressions"]).sum()),
        "invalid_variant": int((~df["variant"].isin(["A", "B"])).sum()),
        "invalid_churn_flag": int((~df["churned_14d"].isin([0, 1])).sum()),
    }
    report["range_checks"] = checks
    bad = (df["avg_session_min"] < 0) | (df["rec_clicks"] > df["rec_impressions"]) | (~df["variant"].isin(["A", "B"]))
    df = df[~bad]

    # 4) Assignment integrity: re-compute the hash bucket for every user
    recomputed = df["user_id"].map(lambda u: assign_variant(u, CONFIG["assignment_salt"], CONFIG["treatment_share"]))
    report["assignment_mismatches"] = int((recomputed != df["variant"]).sum())

    # 5) SRM - overall and per enrollment day
    counts = df["variant"].value_counts()
    report["srm"] = srm_check(counts.get("A", 0), counts.get("B", 0), CONFIG["treatment_share"])
    daily = df.groupby(["enroll_day", "variant"]).size().unstack(fill_value=0)
    daily_p = [srm_check(row["A"], row["B"], CONFIG["treatment_share"])["p_value"] for _, row in daily.iterrows()]
    report["srm_days_flagged"] = int(sum(p < CONFIG["srm_threshold"] for p in daily_p))
    daily.to_csv(RESULTS_DIR / "daily_allocation.csv")

    # 6) Covariate balance (pre-experiment variables must not differ between groups)
    a, b = df[df.variant == "A"], df[df.variant == "B"]
    report["balance_smd"] = {
        "pre_avg_session_min_30d": smd(a["pre_avg_session_min_30d"], b["pre_avg_session_min_30d"]),
        "mobile_share": smd((a["device"] == "mobile").astype(float), (b["device"] == "mobile").astype(float)),
        "enroll_day": smd(a["enroll_day"], b["enroll_day"]),
    }
    report["clean_rows"] = int(len(df))

    df.to_csv(CLEAN_DATA_PATH, index=False)
    with open(RESULTS_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    s = report["srm"]
    print(f"  Raw rows {report['raw_rows']:,}: removed {exact_dups} duplicate rows, "
          f"{conflicting_ids} conflicting ids -> {report['clean_rows']:,} users")
    print(f"  Missing values: {report['missing_values']}  | assignment mismatches: {report['assignment_mismatches']}")
    print(f"  SRM: A={s['n_A']:,} B={s['n_B']:,} share_B={s['share_B']:.4f} chi2={s['chi2']:.2f} "
          f"p={s['p_value']:.3f} -> {'SRM DETECTED - do not trust results' if s['srm_detected'] else 'OK'}")
    print(f"  Balance SMD: " + ", ".join(f"{k}={v:+.3f}" for k, v in report["balance_smd"].items()))
    return df, report
