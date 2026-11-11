"""Run the full project pipeline in the correct order:

    1. Power analysis   -> sample size, MDE, duration
    2. Data simulation  -> data/raw_experiment_data.csv
    3. Data quality     -> duplicates, missing values, SRM, balance -> data/clean_experiment_data.csv
    4. Analysis         -> primary (CUPED, OLS), secondary (BH), guardrails, segments, novelty
    5. Charts           -> outputs/charts/*.png
    6. Report           -> outputs/results/report.md

Usage:  python main.py
"""
import sys

from src import analysis, data_quality, power_analysis, report, simulate_data, visualize
from src.config import CONFIG, ensure_dirs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    print(f"=== {CONFIG['project']} ===")
    print(f"Team: {', '.join(CONFIG['team'])}")

    print("\n[1/6] Power analysis & duration")
    power = power_analysis.run()

    print("\n[2/6] Simulating experiment data")
    simulate_data.run(power["planned_duration_days"])

    print("\n[3/6] Data quality & trust checks")
    clean, quality = data_quality.run()
    if quality["srm"]["srm_detected"]:
        print("  STOP: Sample Ratio Mismatch detected - results are not trustworthy.")
        sys.exit(1)

    print("\n[4/6] Statistical analysis")
    results = analysis.run(clean)

    print("\n[5/6] Charts")
    visualize.run(clean, power, quality, results)

    print("\n[6/6] Decision report")
    report.run(power, quality, results)

    print("\nDone. See outputs/results/report.md and outputs/charts/.")


if __name__ == "__main__":
    main()
