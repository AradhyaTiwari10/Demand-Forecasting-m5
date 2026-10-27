"""
Run the whole of Phase 1:

    python3 run_phase1.py              # uses the prepared data if it already exists
    python3 run_phase1.py --rebuild    # rebuild the prepared data from the raw M5 files

Outputs go to the reports/ folder.
"""

import argparse
from pathlib import Path

import yaml

from backtest import run_backtest, save_backtest_setup
from data_loader import prepare_dataset
from report import build_report


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Kalman filter vs. AML baselines on M5")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rebuild", action="store_true", help="rebuild data/processed from the raw files")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/3: preparing data")
    df = prepare_dataset(config, rebuild=args.rebuild)
    print(str(df["series_id"].nunique()) + " series, " + str(df["date"].min().date()) + " to " + str(df["date"].max().date()))

    print("\nStep 2/3: rolling-origin backtest")
    predictions, origins = run_backtest(df, config)
    predictions.to_csv(reports_dir / "predictions.csv", index=False)
    save_backtest_setup(df, origins, config, reports_dir / "backtest_setup.json")

    print("\nStep 3/3: writing report")
    overall = build_report(df, predictions, origins, config, reports_dir)

    print("\nOverall results (scored days):")
    print(overall[["model", "MAE", "RMSE", "WAPE", "mean_pinball", "coverage"]].round(3).to_string(index=False))
    print("\nDone. See " + str(reports_dir / "phase1_report.md"))


if __name__ == "__main__":
    main()
