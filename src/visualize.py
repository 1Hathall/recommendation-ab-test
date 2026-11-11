"""Step 5 - Charts (saved as PNG in outputs/charts). Labels are in English so they render
on any machine without Arabic font support. Colors follow the course palette:
A = teal, B = orange, reference/true values = navy, naive method = gray."""
import json

import matplotlib

matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CHARTS_DIR, CONFIG, RESULTS_DIR  # noqa: E402
from .power_analysis import power_for_n  # noqa: E402

TEAL, ORANGE, NAVY, GRAY, RED = "#0E7490", "#E58A1F", "#0F2A3F", "#A9B4BC", "#B42318"
plt.rcParams.update({
    "figure.figsize": (9, 4.8), "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.edgecolor": "#56656F", "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": "#3D4F5C", "ytick.color": "#3D4F5C",
    "axes.grid": True, "grid.color": "#E3E9ED", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 10.5, "legend.frameon": False,
})


def _save(fig, name):
    fig.savefig(CHARTS_DIR / name)
    plt.close(fig)


def power_curve(power: dict):
    c = CONFIG
    n = np.linspace(500, 12_000, 300)
    fig, ax = plt.subplots()
    for rel, color, ls in [(0.05, TEAL, "-"), (0.08, ORANGE, "-"), (0.15, NAVY, "--")]:
        ax.plot(n, power_for_n(c["baseline_session_sd"], c["baseline_session_min"] * rel, n, c["alpha"]),
                color=color, lw=2, ls=ls, label=f"true lift = {rel:.0%}")
    ax.axhline(c["power"], color=GRAY, lw=1, ls=":")
    ax.axvline(power["n_per_group_required"], color=TEAL, lw=1, ls=":")
    ax.axvline(power["planned_n_per_group"], color=NAVY, lw=1, ls=":")
    ax.annotate(f"required n = {power['n_per_group_required']:,}", (power["n_per_group_required"], 0.3),
                xytext=(8, 0), textcoords="offset points", fontsize=9)
    ax.annotate(f"planned n = {power['planned_n_per_group']:,}\n({power['planned_duration_days']} days)",
                (power["planned_n_per_group"], 0.15), xytext=(8, 0), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Users per group", ylabel="Power", ylim=(0, 1.02),
           title="Power vs. sample size (alpha = 0.05, SD = 16 min)")
    ax.legend(loc="lower right")
    _save(fig, "01_power_curve.png")


def allocation(df: pd.DataFrame, quality: dict):
    daily = df.groupby(["enroll_day", "variant"]).size().unstack()
    fig, ax = plt.subplots()
    x = daily.index.values
    ax.bar(x - 0.2, daily["A"], width=0.38, color=TEAL, label="A (control)")
    ax.bar(x + 0.2, daily["B"], width=0.38, color=ORANGE, label="B (new recommendations)")
    s = quality["srm"]
    ax.set(xlabel="Enrollment day", ylabel="Users", xticks=x,
           title=f"Daily allocation - SRM chi2 = {s['chi2']:.2f}, p = {s['p_value']:.2f} (no mismatch)")
    ax.legend(loc="lower right")
    _save(fig, "02_daily_allocation_srm.png")


def distribution(df: pd.DataFrame):
    fig, ax = plt.subplots()
    bins = np.logspace(np.log10(1), np.log10(200), 45)
    for v, color, lab in [("A", TEAL, "A (control)"), ("B", ORANGE, "B (new recommendations)")]:
        vals = df.loc[df.variant == v, "avg_session_min"]
        ax.hist(vals, bins=bins, histtype="step", lw=2, color=color, label=f"{lab}  mean={vals.mean():.1f}")
    ax.set_xscale("log")
    ax.set(xlabel="Average session duration per user (minutes, log scale)", ylabel="Users",
           title="Distribution of the primary metric (right-skewed)")
    ax.legend()
    _save(fig, "03_session_distribution.png")


def primary_estimates(res: dict, truth: dict):
    p = res["primary"]
    rows = [("Difference in means", p["unadjusted"], GRAY),
            ("CUPED (pre-period)", p["cuped"], TEAL),
            ("OLS + covariates (HC1)", p["regression"], ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (lab, r, color) in enumerate(rows):
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                    fmt="o", color=color, ms=8, capsize=5, lw=2)
        ax.annotate(f"{r['diff']:+.2f}  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", (r["ci_high"], i),
                    xytext=(8, -4), textcoords="offset points", fontsize=9)
    mde = CONFIG["baseline_session_min"] * CONFIG["mde_rel"]
    ax.axvline(0, color=NAVY, lw=1)
    ax.axvline(mde, color=GRAY, lw=1, ls="--")
    ax.axvline(truth["true_effect_min_population"], color=NAVY, lw=1.5, ls=":")
    ax.text(mde, 2.45, " MDE", fontsize=9, color="#56656F")
    ax.text(truth["true_effect_min_population"], -0.55, " true effect (simulation)", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlim(-0.3, max(r["ci_high"] for _, r, _ in rows) + 1.3)
    ax.set(xlabel="Effect on avg. session duration, B - A (minutes), 95% CI",
           title="Primary metric: estimated effect of personalized recommendations")
    _save(fig, "04_primary_effect_ci.png")


def group_means(res: dict):
    p = res["primary"]["unadjusted"]
    se_a = (p["ci_high"] - p["ci_low"]) / (2 * 1.96) / np.sqrt(2)  # approx per-group SE for display
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.grid(axis="x", visible=False)
    ax.bar(["A (control)", "B (new recs)"], [p["mean_A"], p["mean_B"]], color=[TEAL, ORANGE], width=0.5,
           yerr=[1.96 * se_a] * 2, capsize=6)
    for i, m in enumerate([p["mean_A"], p["mean_B"]]):
        ax.text(i, m / 2, f"{m:.2f} min", ha="center", color="white", fontweight="bold")
    ax.set(ylabel="Avg. session duration (minutes)",
           title=f"Mean session duration by group (lift {p['rel_lift']:+.1%})")
    _save(fig, "05_group_means.png")


def secondary_and_guardrails(res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sec = res["secondary"]
    names = list(sec)
    ax = axes[0]
    for i, k in enumerate(names):
        r = sec[k]
        color = ORANGE if r["significant_after_bh"] else GRAY
        ax.errorbar(r["rel_lift"] * 100, i, xerr=[[(r["rel_lift"] - r["rel_ci_low"]) * 100],
                                                  [(r["rel_ci_high"] - r["rel_lift"]) * 100]],
                    fmt="o", color=color, ms=8, capsize=5, lw=2)
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set(xlabel="Relative lift B vs A (%), 95% CI", title="Secondary metrics (BH-corrected)")

    ax = axes[1]
    g = res["guardrails"]
    items = [("Churn rate 14d (pp)", g["churn_rate_14d"], 100), ("Avg latency (ms)", g["avg_latency_ms"], 1)]
    for i, (lab, r, k) in enumerate(items):
        color = TEAL if r["ok"] else RED
        ax.errorbar(r["diff"] * k / (r["margin"] * k), i,
                    xerr=[[(r["diff"] - r["ci_low"]) / r["margin"]], [(r["ci_high"] - r["diff"]) / r["margin"]]],
                    fmt="o", color=color, ms=8, capsize=5, lw=2)
        ax.annotate(f"{r['diff'] * k:+.2f} (upper {r['ci_high'] * k:+.2f}, margin +{r['margin'] * k:g})",
                    (r["diff"] / r["margin"], i), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9)
    ax.axvline(0, color=NAVY, lw=1)
    ax.axvline(1, color=RED, lw=1.2, ls="--")
    ax.text(1, 1.45, " margin", color=RED, fontsize=9)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.set_ylim(-0.6, 1.7)
    ax.set(xlabel="Change B - A as a fraction of the allowed margin", title="Guardrails (non-inferiority)")
    fig.tight_layout()
    _save(fig, "06_secondary_guardrails.png")


def daily_effect(res: dict):
    d = pd.read_csv(RESULTS_DIR / "daily_effect.csv")
    fig, ax = plt.subplots()
    ax.fill_between(d["enroll_day"], d["ci_low"], d["ci_high"], color=ORANGE, alpha=0.18, lw=0, label="95% CI")
    ax.plot(d["enroll_day"], d["diff"], color=ORANGE, lw=2, marker="o", ms=5, label="Daily effect B - A")
    ax.axhline(res["primary"]["unadjusted"]["diff"], color=NAVY, lw=1.2, ls="--", label="Overall effect")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(xlabel="Enrollment day", ylabel="Effect (minutes)", xticks=d["enroll_day"],
           title="Effect by enrollment day - novelty check")
    ax.legend(loc="upper right")
    _save(fig, "07_daily_effect_novelty.png")


def segments(res: dict):
    seg = res["segments"]["by_device"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (k, r) in enumerate(seg.items()):
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                    fmt="o", color=TEAL if k == "mobile" else ORANGE, ms=8, capsize=5, lw=2)
        ax.annotate(f"{r['diff']:+.2f} min ({r['rel_lift']:+.1%}), n={r['n_A'] + r['n_B']:,}",
                    (r["ci_high"], i), xytext=(8, -4), textcoords="offset points", fontsize=9)
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_yticks(range(len(seg)), list(seg))
    ax.set_ylim(-0.6, len(seg) - 0.4)
    ax.set_xlim(right=max(r["ci_high"] for r in seg.values()) + 1.6)
    ax.set(xlabel="Effect B - A (minutes), 95% CI",
           title=f"Effect by device (exploratory; interaction p = {res['segments']['interaction_p_value']:.2f})")
    _save(fig, "08_segments_device.png")


def naive_vs_experiment(res: dict, truth: dict):
    nv = res["naive_observational"]
    vals = [nv["clickers_vs_nonclickers_diff"], nv["adjusted_for_pre_period_diff"],
            nv["experimental_effect"], truth["true_effect_min_population"]]
    labels = ["Naive: clickers vs\nnon-clickers", "Naive, adjusted\nfor pre-period", "Randomized A/B\nestimate",
              "True effect\n(simulation)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.grid(axis="x", visible=False)
    bars = ax.bar(labels, vals, color=[GRAY, GRAY, ORANGE, NAVY], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=10)
    ax.set(ylabel="Estimated effect (minutes)", title="Why randomization? Self-selection inflates naive estimates")
    _save(fig, "09_naive_vs_experiment.png")


def run(df: pd.DataFrame, power: dict, quality: dict, res: dict):
    with open(RESULTS_DIR / "simulation_truth.json", encoding="utf-8") as f:
        truth = json.load(f)
    power_curve(power)
    allocation(df, quality)
    distribution(df)
    primary_estimates(res, truth)
    group_means(res)
    secondary_and_guardrails(res)
    daily_effect(res)
    segments(res)
    naive_vs_experiment(res, truth)
    print(f"  Saved {len(list(CHARTS_DIR.glob('*.png')))} charts -> outputs/charts/")
