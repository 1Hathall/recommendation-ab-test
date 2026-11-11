"""Step 1 - Sample size, minimum detectable effect (MDE) and experiment duration.

Primary metric is continuous (average session duration per user), so we use the
two-sample formula for a difference in means:

    n per group = 2 * (z_{1-alpha/2} + z_{power})^2 * sigma^2 / delta^2
"""
import json

import numpy as np
from scipy import stats

from .config import CONFIG, RESULTS_DIR


def n_per_group_means(sd: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (z_a + z_b) ** 2 * sd ** 2 / delta ** 2))


def mde_means(sd: float, n_per_group: int, alpha: float = 0.05, power: float = 0.80) -> float:
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((z_a + z_b) * sd * np.sqrt(2 / n_per_group))


def power_for_n(sd: float, delta: float, n_per_group, alpha: float = 0.05):
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = sd * np.sqrt(2 / np.asarray(n_per_group, dtype=float))
    return stats.norm.cdf(delta / se - z_a)


def duration_days(n_per_group: int, daily_users: int, groups: int = 2, min_weeks: int = 1) -> int:
    days = int(np.ceil(n_per_group * groups / daily_users))
    weeks = max(min_weeks, int(np.ceil(days / 7)))
    return weeks * 7


def run() -> dict:
    c = CONFIG
    delta = c["baseline_session_min"] * c["mde_rel"]
    n_req = n_per_group_means(c["baseline_session_sd"], delta, c["alpha"], c["power"])
    days = duration_days(n_req, c["daily_new_users"], min_weeks=c["min_weeks"])
    n_planned = days * c["daily_new_users"] // 2
    mde_planned = mde_means(c["baseline_session_sd"], n_planned, c["alpha"], c["power"])
    mde_one_week = mde_means(c["baseline_session_sd"], 7 * c["daily_new_users"] // 2, c["alpha"], c["power"])
    # If CUPED removes ~36% of variance (pre/post correlation ~0.6), the required n shrinks accordingly.
    assumed_rho = 0.6
    n_req_cuped = n_per_group_means(c["baseline_session_sd"] * np.sqrt(1 - assumed_rho ** 2), delta,
                                    c["alpha"], c["power"])
    n_hyp = n_per_group_means(c["baseline_session_sd"], c["baseline_session_min"] * c["hypothesized_lift_rel"],
                              c["alpha"], c["power"])

    result = {
        "alpha": c["alpha"],
        "power": c["power"],
        "baseline_mean_min": c["baseline_session_min"],
        "baseline_sd_min": c["baseline_session_sd"],
        "mde_rel": c["mde_rel"],
        "mde_abs_min": delta,
        "n_per_group_required": n_req,
        "n_total_required": 2 * n_req,
        "n_per_group_required_if_cuped_rho_0.6": n_req_cuped,
        "n_per_group_to_detect_hypothesized_15pct": n_hyp,
        "daily_new_users": c["daily_new_users"],
        "raw_days_needed": int(np.ceil(2 * n_req / c["daily_new_users"])),
        "planned_duration_days": days,
        "planned_n_per_group": n_planned,
        "mde_at_planned_n_min": mde_planned,
        "mde_at_planned_n_rel": mde_planned / c["baseline_session_min"],
        "mde_if_only_one_week_min": mde_one_week,
        "power_at_planned_n_for_mde": float(power_for_n(c["baseline_session_sd"], delta, n_planned, c["alpha"])),
    }
    with open(RESULTS_DIR / "power_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  MDE = {c['mde_rel']:.0%} relative = {delta:.2f} min  (alpha={c['alpha']}, power={c['power']:.0%})")
    print(f"  Required sample size: {n_req:,} per group ({2 * n_req:,} total)")
    print(f"  Daily users {c['daily_new_users']:,} -> {result['raw_days_needed']} days needed, "
          f"rounded to {days} days ({days // 7} full weeks)")
    print(f"  Planned n per group: {n_planned:,} -> detectable effect {mde_planned:.2f} min "
          f"({result['mde_at_planned_n_rel']:.1%}), power for MDE = {result['power_at_planned_n_for_mde']:.1%}")
    return result
