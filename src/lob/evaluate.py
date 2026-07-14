"""Walk-forward OLS on the binned features.

Chronological folds: train on fold i, test on fold i+1, standardisation fit
on the training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    r2_os: float
    corr: float


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def _ols_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def walk_forward(df: pd.DataFrame, feature_cols: list[str], target_col: str,
                 n_folds: int = 12) -> tuple[list[FoldResult], pd.Series]:
    """Adjacent-fold walk-forward OLS. Returns per-fold stats and pooled OOS predictions."""
    idx = np.array_split(np.arange(len(df)), n_folds)
    X_all = df[feature_cols].to_numpy()
    y_all = df[target_col].to_numpy()

    results, preds = [], []
    for i in range(n_folds - 1):
        tr, te = idx[i], idx[i + 1]
        mu = X_all[tr].mean(axis=0)
        sd = X_all[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (X_all[tr] - mu) / sd, (X_all[te] - mu) / sd
        beta = _ols_fit(Xtr, y_all[tr])
        yhat = _ols_predict(Xte, beta)

        bench = y_all[tr].mean()
        sse = np.sum((y_all[te] - yhat) ** 2)
        sst = np.sum((y_all[te] - bench) ** 2)
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        corr = (np.corrcoef(yhat, y_all[te])[0, 1]
                if yhat.std() > 0 and y_all[te].std() > 0 else np.nan)
        results.append(FoldResult(i + 1, len(tr), len(te), r2, corr))
        preds.append(pd.Series(yhat, index=df.index[te]))

    return results, pd.concat(preds)


def pooled_r2(df: pd.DataFrame, preds: pd.Series, target_col: str) -> float:
    """Pooled OOS R^2 against a zero forecast (1s returns have ~0 mean, and
    zero is stricter than per-fold training means)."""
    y = df.loc[preds.index, target_col].to_numpy()
    yhat = preds.to_numpy()
    sst = np.sum(y ** 2)
    return 1.0 - np.sum((y - yhat) ** 2) / sst if sst > 0 else np.nan
