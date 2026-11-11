"""Step 4 - Statistical analysis (pre-registered plan, see README section "Analysis plan").

Primary metric : average session duration per user (minutes), B - A
    * Welch difference in means with 95% CI + p-value (normal approximation)
    * relative lift with delta-method CI
    * CUPED using the 30-day pre-period session duration (variance reduction)
    * OLS regression with covariates and robust (HC1) standard errors (robustness check)
    * Mann-Whitney U test (robustness to skewness)
Secondary      : recommendation CTR (ratio metric, delta method), items viewed per session,
                 sessions per user -> Benjamini-Hochberg correction for multiple testing
Guardrails     : 14-day churn rate, mean server latency -> non-inferiority vs pre-set margins
Exploratory    : effect by device (interaction test), effect over time (novelty check),
                 and a "naive observational" estimate to show why randomization matters.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import CONFIG, RESULTS_DIR

Z = stats.norm.ppf(0.975)


def diff_ci(a, b) -> dict:
    """Difference in means (B - A) with 95% CI and two-sided p-value (Welch / normal approx)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff, se = mb - ma, np.sqrt(va + vb)
    # relative lift via delta method: var(mb/ma) ~ vb/ma^2 + mb^2 va / ma^4
    rel = diff / ma
    se_rel = np.sqrt(vb / ma ** 2 + mb ** 2 * va / ma ** 4)
    return {"mean_A": ma, "mean_B": mb, "diff": diff, "ci_low": diff - Z * se, "ci_high": diff + Z * se,
            "se": se, "p_value": float(2 * stats.norm.sf(abs(diff / se))),
            "rel_lift": rel, "rel_ci_low": rel - Z * se_rel, "rel_ci_high": rel + Z * se_rel,
            "n_A": len(a), "n_B": len(b)}


def ratio_diff_ci(num_a, den_a, num_b, den_b) -> dict:
    """Ratio metric (sum num / sum den) per group with delta-method variance; users are the unit."""
    def ratio_and_var(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        n, mx, my = len(num), den.mean(), num.mean()
        r = my / mx
        cov = np.cov(num, den, ddof=1)
        var = (cov[0, 0] - 2 * r * cov[0, 1] + r ** 2 * cov[1, 1]) / (n * mx ** 2)
        return r, var
    ra, va = ratio_and_var(num_a, den_a)
    rb, vb = ratio_and_var(num_b, den_b)
    diff, se = rb - ra, np.sqrt(va + vb)
    se_rel = np.sqrt(vb / ra ** 2 + rb ** 2 * va / ra ** 4)
    return {"mean_A": ra, "mean_B": rb, "diff": diff, "ci_low": diff - Z * se, "ci_high": diff + Z * se,
            "se": se, "p_value": float(2 * stats.norm.sf(abs(diff / se))), "rel_lift": diff / ra,
            "rel_ci_low": diff / ra - Z * se_rel, "rel_ci_high": diff / ra + Z * se_rel,
            "n_A": len(num_a), "n_B": len(num_b)}


def cuped(df: pd.DataFrame, y: str, x: str) -> tuple[pd.Series, float, float]:
    theta = np.cov(df[y], df[x], ddof=1)[0, 1] / df[x].var(ddof=1)
    adjusted = df[y] - theta * (df[x] - df[x].mean())
    rho = df[[y, x]].corr().iloc[0, 1]
    return adjusted, float(theta), float(rho)


def decide(ci_low: float, ci_high: float, mde: float, guardrails_ok: bool) -> tuple[str, str]:
    """Decision rule written BEFORE looking at the data (same rule as the course notebook)."""
    if ci_high < 0:
        return "DO NOT SHIP", "لا نطلق، لأن النسخة الجديدة أضرّت بالمقياس الرئيسي."
    if ci_low > 0 and not guardrails_ok:
        return ("DO NOT SHIP YET", "لا نطلق الآن: المقياس الرئيسي تحسّن لكن أحد مقاييس الحماية ساء. "
                "نعالج المشكلة ثم نعيد التحقق.")
    if ci_low > 0:
        return "SHIP", "نطلق، لأن الأثر إيجابي بثقة ومقاييس الحماية سليمة."
    if ci_high < mde:
        return "DO NOT SHIP", "لا نطلق، لأننا استبعدنا وجود أثر بالحجم الذي يهمنا."
    return "INCONCLUSIVE", "النتيجة غير حاسمة: نمدّد التجربة أو نعيد تصميمها بقوة إحصائية أعلى."


def run(df: pd.DataFrame) -> dict:
    c = CONFIG
    df = df.copy()
    df["treat"] = (df["variant"] == "B").astype(int)
    A, B = df[df.treat == 0], df[df.treat == 1]
    res: dict = {}

    # ---------------- Primary metric ----------------
    primary = diff_ci(A["avg_session_min"], B["avg_session_min"])
    adj, theta, rho = cuped(df, "avg_session_min", "pre_avg_session_min_30d")
    primary_cuped = diff_ci(adj[df.treat == 0], adj[df.treat == 1])
    primary_cuped["rel_lift"] = primary_cuped["diff"] / primary["mean_A"]
    primary_cuped["rel_ci_low"] = primary_cuped["ci_low"] / primary["mean_A"]
    primary_cuped["rel_ci_high"] = primary_cuped["ci_high"] / primary["mean_A"]
    var_reduction = 1 - (primary_cuped["se"] / primary["se"]) ** 2

    ols = smf.ols("avg_session_min ~ treat + pre_avg_session_min_30d + C(device) + C(enroll_day)",
                  data=df).fit(cov_type="HC1")
    ci = ols.conf_int().loc["treat"]
    regression = {"diff": float(ols.params["treat"]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                  "se": float(ols.bse["treat"]), "p_value": float(ols.pvalues["treat"]), "r_squared": float(ols.rsquared)}
    mw = stats.mannwhitneyu(B["avg_session_min"], A["avg_session_min"], alternative="two-sided")

    res["primary"] = {"metric": "avg_session_min", "unadjusted": primary, "cuped": primary_cuped,
                      "cuped_theta": theta, "pre_post_correlation": rho, "cuped_variance_reduction": var_reduction,
                      "regression": regression, "mann_whitney_p": float(mw.pvalue),
                      "median_A": float(A["avg_session_min"].median()), "median_B": float(B["avg_session_min"].median())}
    (ols.summary2().tables[1]).to_csv(RESULTS_DIR / "regression_coefficients.csv")

    # ---------------- Secondary metrics + BH correction ----------------
    secondary = {
        "recommendation_ctr": ratio_diff_ci(A["rec_clicks"], A["rec_impressions"], B["rec_clicks"], B["rec_impressions"]),
        "items_viewed_per_session": diff_ci(A["items_viewed_per_session"], B["items_viewed_per_session"]),
        "sessions_per_user_7d": diff_ci(A["sessions_7d"], B["sessions_7d"]),
    }
    pvals = [v["p_value"] for v in secondary.values()]
    reject, p_adj, _, _ = multipletests(pvals, alpha=c["alpha"], method="fdr_bh")
    for (k, v), rj, pa in zip(secondary.items(), reject, p_adj):
        v["p_value_bh"] = float(pa)
        v["significant_after_bh"] = bool(rj)
    res["secondary"] = secondary

    # ---------------- Guardrails (non-inferiority) ----------------
    churn = diff_ci(A["churned_14d"], B["churned_14d"])
    lat_a, lat_b = A["avg_latency_ms"].dropna(), B["avg_latency_ms"].dropna()
    latency = diff_ci(lat_a, lat_b)
    churn["margin"], latency["margin"] = c["churn_margin_abs"], c["latency_margin_ms"]
    churn["ok"] = bool(churn["ci_high"] < c["churn_margin_abs"])
    latency["ok"] = bool(latency["ci_high"] < c["latency_margin_ms"])
    latency["p95_A"], latency["p95_B"] = float(lat_a.quantile(0.95)), float(lat_b.quantile(0.95))
    res["guardrails"] = {"churn_rate_14d": churn, "avg_latency_ms": latency}
    guardrails_ok = churn["ok"] and latency["ok"]

    # ---------------- Exploratory: segments, novelty ----------------
    seg = {}
    for dev, g in df.groupby("device"):
        seg[dev] = diff_ci(g.loc[g.treat == 0, "avg_session_min"], g.loc[g.treat == 1, "avg_session_min"])
    inter = smf.ols("avg_session_min ~ treat * C(device) + pre_avg_session_min_30d", data=df).fit(cov_type="HC1")
    inter_term = [k for k in inter.params.index if k.startswith("treat:")][0]
    res["segments"] = {"by_device": seg, "interaction_p_value": float(inter.pvalues[inter_term])}

    daily = []
    for d, g in df.groupby("enroll_day"):
        r = diff_ci(g.loc[g.treat == 0, "avg_session_min"], g.loc[g.treat == 1, "avg_session_min"])
        daily.append({"enroll_day": int(d), **{k: r[k] for k in ("diff", "ci_low", "ci_high", "rel_lift")}})
    pd.DataFrame(daily).to_csv(RESULTS_DIR / "daily_effect.csv", index=False)
    w1 = df[df.enroll_day <= 7]
    w2 = df[df.enroll_day > 7]
    res["novelty"] = {
        "week1": diff_ci(w1.loc[w1.treat == 0, "avg_session_min"], w1.loc[w1.treat == 1, "avg_session_min"]),
        "week2": diff_ci(w2.loc[w2.treat == 0, "avg_session_min"], w2.loc[w2.treat == 1, "avg_session_min"]),
    }

    # ---------------- Why randomization? naive observational comparison ----------------
    # Inside group B, compare users who clicked recommendations vs. those who did not.
    # Clicking is self-selected and driven by engagement (confounder) -> biased "effect".
    clicked = B["rec_clicks"] > 0
    naive = diff_ci(B.loc[~clicked, "avg_session_min"], B.loc[clicked, "avg_session_min"])
    naive_adj = smf.ols("avg_session_min ~ clicked + pre_avg_session_min_30d",
                        data=B.assign(clicked=clicked.astype(int))).fit(cov_type="HC1")
    res["naive_observational"] = {
        "clickers_vs_nonclickers_diff": naive["diff"], "share_clickers": float(clicked.mean()),
        "adjusted_for_pre_period_diff": float(naive_adj.params["clicked"]),
        "experimental_effect": primary["diff"],
    }

    # ---------------- Decision ----------------
    mde_abs = c["baseline_session_min"] * c["mde_rel"]
    decision_en, decision_ar = decide(primary["ci_low"], primary["ci_high"], mde_abs, guardrails_ok)
    extra_min_per_user = primary["diff"]
    res["decision"] = {
        "decision": decision_en, "decision_ar": decision_ar, "guardrails_ok": guardrails_ok,
        "mde_abs_min": mde_abs, "primary_ci_low_above_zero": bool(primary["ci_low"] > 0),
        "primary_ci_low_above_mde": bool(primary["ci_low"] > mde_abs),
        "hypothesized_15pct_inside_ci": bool(primary["rel_ci_low"] <= c["hypothesized_lift_rel"] <= primary["rel_ci_high"]),
        "annual_users": c["annual_users"],
        # rough business translation: extra minutes per session x sessions in the first week,
        # summed over the yearly new users (each counted for their first week only)
        "extra_min_per_user_first_week": extra_min_per_user * float(B["sessions_7d"].mean()),
        "extra_hours_first_week_all_annual_users":
            extra_min_per_user * float(B["sessions_7d"].mean()) * c["annual_users"] / 60,
    }

    with open(RESULTS_DIR / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    _summary_table(res).to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    p, pc = primary, primary_cuped
    print(f"  Primary  A={p['mean_A']:.2f} min  B={p['mean_B']:.2f} min  diff={p['diff']:+.2f} "
          f"[{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]  lift={p['rel_lift']:+.1%}  p={p['p_value']:.2g}")
    print(f"  CUPED    diff={pc['diff']:+.2f} [{pc['ci_low']:+.2f}, {pc['ci_high']:+.2f}]  "
          f"variance reduction={var_reduction:.0%} (rho={rho:.2f})")
    print(f"  OLS      diff={regression['diff']:+.2f} [{regression['ci_low']:+.2f}, {regression['ci_high']:+.2f}]")
    for k, v in secondary.items():
        print(f"  {k:<26} lift={v['rel_lift']:+.1%}  p_BH={v['p_value_bh']:.2g}  sig={v['significant_after_bh']}")
    print(f"  Guardrail churn   diff={churn['diff']*100:+.2f}pp upper={churn['ci_high']*100:+.2f}pp "
          f"(margin +{c['churn_margin_abs']*100:.1f}pp) -> {'OK' if churn['ok'] else 'FAIL'}")
    print(f"  Guardrail latency diff={latency['diff']:+.1f}ms upper={latency['ci_high']:+.1f}ms "
          f"(margin +{c['latency_margin_ms']:.0f}ms) -> {'OK' if latency['ok'] else 'FAIL'}")
    print(f"  DECISION: {decision_en}")
    return res


def _summary_table(res: dict) -> pd.DataFrame:
    rows = []
    def add(group, name, r):
        rows.append({"type": group, "metric": name, "control_A": r["mean_A"], "treatment_B": r["mean_B"],
                     "diff_B_minus_A": r["diff"], "ci95_low": r["ci_low"], "ci95_high": r["ci_high"],
                     "relative_lift": r.get("rel_lift"), "p_value": r["p_value"],
                     "p_value_bh": r.get("p_value_bh"), "status": r.get("status", "")})
    add("primary", "avg_session_min", res["primary"]["unadjusted"])
    add("primary (CUPED)", "avg_session_min", res["primary"]["cuped"])
    for k, v in res["secondary"].items():
        v = dict(v, status="significant" if v["significant_after_bh"] else "not significant")
        add("secondary", k, v)
    for k, v in res["guardrails"].items():
        v = dict(v, status="OK" if v["ok"] else "FAIL")
        add("guardrail", k, v)
    return pd.DataFrame(rows)
