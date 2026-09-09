"""Load the LOBSTER day, build the features, run the walk-forward, write results/."""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lob.data import load_day
from lob.features import build_dataset
from lob.evaluate import walk_forward, pooled_r2

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

TICKERS = ["MSFT", "INTC", "AMZN"]
DAY = "2012-06-21"
SESSION = "34200000_57600000"
HORIZONS = (1, 2, 5, 10, 30, 60)
FEATURES_FULL = ["ofi", "tflow", "qi", "depth_imb"]
FEATURES_OFI = ["ofi"]
N_FOLDS = 12


def main():
    rows = []
    for tic in TICKERS:
        msg = os.path.join(DATA, f"{tic}_{DAY}_{SESSION}_message_10.csv")
        obk = os.path.join(DATA, f"{tic}_{DAY}_{SESSION}_orderbook_10.csv")
        print(f"[{tic}] loading ...", flush=True)
        book = load_day(msg, obk, levels=10)
        print(f"[{tic}] {len(book):,} events; building 1s dataset ...", flush=True)
        df = build_dataset(book, bin_seconds=1.0, horizons=HORIZONS)
        med_spread_bps = float(np.median(1e4 * df["spread"] / df["mid"]))
        print(f"[{tic}] {len(df):,} bins; median spread {med_spread_bps:.2f} bps", flush=True)

        for h in HORIZONS:
            target = f"ret_{h}s"
            for name, feats in [("full", FEATURES_FULL), ("ofi_only", FEATURES_OFI)]:
                folds, preds = walk_forward(df, feats, target, n_folds=N_FOLDS)
                r2_pool = pooled_r2(df, preds, target)
                r2_folds = np.array([f.r2_os for f in folds])
                corr = np.nanmean([f.corr for f in folds])
                # economic relevance: how often is the predicted move bigger
                # than half the concurrent relative spread?
                half_spread_bps = 1e4 * df.loc[preds.index, "spread"] / (2 * df.loc[preds.index, "mid"])
                frac_tradeable = float((preds.abs() > half_spread_bps).mean())
                rows.append({
                    "ticker": tic, "horizon_s": h, "model": name,
                    "r2_os_pooled": r2_pool,
                    "r2_os_fold_mean": float(np.nanmean(r2_folds)),
                    "r2_os_fold_min": float(np.nanmin(r2_folds)),
                    "r2_os_fold_max": float(np.nanmax(r2_folds)),
                    "pred_target_corr": float(corr),
                    "frac_pred_gt_half_spread": frac_tradeable,
                    "n_bins": len(df),
                    "median_spread_bps": med_spread_bps,
                })
        print(f"[{tic}] done", flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tic in TICKERS:
        sub = res[(res.ticker == tic) & (res.model == "full")]
        ax.plot(sub["horizon_s"], 100 * sub["r2_os_pooled"], marker="o", label=tic)
    ax.set_xscale("log")
    ax.set_xlabel("forecast horizon (seconds, log scale)")
    ax.set_ylabel("out-of-sample $R^2$ (%)")
    ax.set_title("Signal decay: walk-forward OOS $R^2$ of 1s LOB features\n"
                 "(OFI + trade flow + queue/depth imbalance, OLS, 2012-06-21)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "signal_decay.png"), dpi=150)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
