"""Step 2 - Generate the experiment dataset by simulation.

IMPORTANT: the data is SIMULATED. In a simulation we choose the true effect ourselves
and then check whether the analysis recovers it. The assumptions are in config.CONFIG.

One row = one user. Users enter during the enrollment window (planned duration) and each
user is observed for a fixed 7-day window after first exposure, so early and late
entrants are comparable.

Data-generating story (also the causal DAG used in the README):
    engagement (latent)  -> pre-period session time, sessions, CTR, churn, outcome
    variant (random)     -> outcome, CTR, items, latency, churn
Because variant is assigned by a hash of user_id, it is independent of engagement,
so a simple difference in means is an unbiased estimate of the causal effect.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from .config import CONFIG, RAW_DATA_PATH, RESULTS_DIR


def assign_variant(user_id: str, salt: str, treatment_share: float = 0.5) -> str:
    """Deterministic hash-based randomization: same user always gets the same variant."""
    bucket = int(hashlib.md5(f"{salt}:{user_id}".encode("utf-8")).hexdigest(), 16) % 10_000
    return "B" if bucket < treatment_share * 10_000 else "A"


def simulate(cfg: dict, duration_days: int) -> tuple[pd.DataFrame, dict]:
    r = np.random.default_rng(cfg["seed"])
    n = duration_days * cfg["daily_new_users"]

    user_id = np.array([f"U{i:06d}" for i in range(1, n + 1)])
    variant = np.array([assign_variant(u, cfg["assignment_salt"], cfg["treatment_share"]) for u in user_id])
    is_b = (variant == "B").astype(int)
    day = r.integers(1, duration_days + 1, n)
    device = np.where(r.random(n) < cfg["mobile_share"], "mobile", "web")
    engagement = r.normal(0, 1, n)  # latent user engagement (not observed directly)

    # Session duration: log-normal, mean ~20 min, SD ~16 min. Pre & post share the engagement term.
    mu = np.log(cfg["baseline_session_min"]) - 0.248
    pre_session = np.exp(mu + 0.55 * engagement + r.normal(0, 0.44, n))
    base_session = np.exp(mu + 0.55 * engagement + r.normal(0, 0.44, n))

    # Heterogeneous, time-varying treatment effect, normalised so its average equals true_lift_rel
    device_mult = np.where(device == "mobile", cfg["mobile_lift_multiplier"], cfg["web_lift_multiplier"])
    novelty = 1 + cfg["novelty_boost"] * np.exp(-(day - 1) / cfg["novelty_decay_days"])
    shape = device_mult * novelty
    lift_u = cfg["true_lift_rel"] * shape / shape.mean()
    avg_session = base_session * (1 + lift_u * is_b)

    sessions = 1 + r.poisson(3 * np.exp(0.3 * engagement))
    items = np.maximum(0.5, cfg["items_per_session"] * np.exp(0.15 * engagement)
                       * (1 + cfg["items_effect_rel"] * is_b) * r.gamma(20, 1 / 20, n))
    impressions = r.poisson(8 * sessions)
    ctr_u = np.clip(cfg["base_ctr"] * np.exp(0.35 * engagement - 0.06) * (1 + cfg["ctr_effect_rel"] * is_b), 0, 1)
    clicks = r.binomial(impressions, ctr_u)
    churn_base = cfg["churn_rate"] * np.exp(-0.5 * engagement)
    churn_base = churn_base * cfg["churn_rate"] / churn_base.mean()
    churned = r.binomial(1, np.clip(churn_base + cfg["churn_effect_abs"] * is_b, 0, 1))
    latency = np.maximum(50, r.normal(cfg["latency_mean_ms"] + cfg["latency_effect_ms"] * is_b,
                                      cfg["latency_sd_ms"], n))

    df = pd.DataFrame({
        "user_id": user_id,
        "variant": variant,
        "enroll_day": day,
        "device": device,
        "pre_avg_session_min_30d": pre_session.round(2),
        "avg_session_min": avg_session.round(2),
        "sessions_7d": sessions,
        "items_viewed_per_session": items.round(2),
        "rec_impressions": impressions,
        "rec_clicks": clicks,
        "churned_14d": churned,
        "avg_latency_ms": latency.round(1),
    })

    # Realistic data problems to be caught by the quality checks
    miss = r.random(n) < cfg["missing_latency_share"]
    df.loc[miss, "avg_latency_ms"] = np.nan
    dup_idx = r.choice(n, int(n * cfg["duplicate_share"]), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    truth = {
        "note": "Ground truth of the simulation - unknown to an analyst in a real experiment.",
        "true_lift_rel": cfg["true_lift_rel"],
        "true_effect_min_population": float(np.mean(base_session * lift_u)),
        "true_ctr_effect_rel": cfg["ctr_effect_rel"],
        "true_items_effect_rel": cfg["items_effect_rel"],
        "true_churn_effect_abs": cfg["churn_effect_abs"],
        "true_latency_effect_ms": cfg["latency_effect_ms"],
        "novelty_boost": cfg["novelty_boost"],
        "injected_duplicate_rows": int(len(dup_idx)),
        "injected_missing_latency": int(miss.sum()),
    }
    return df, truth


def run(duration_days: int) -> pd.DataFrame:
    df, truth = simulate(CONFIG, duration_days)
    df.to_csv(RAW_DATA_PATH, index=False)
    with open(RESULTS_DIR / "simulation_truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    print(f"  Simulated {len(df):,} rows over {duration_days} enrollment days -> {RAW_DATA_PATH.name}")
    return df
