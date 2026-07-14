"""Event-level microstructure features and time-binned aggregation.

All features at bin t use information from events with time <= the bin's right
edge only; targets are built strictly from later bin edges (see build_dataset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_ofi(bid_p: np.ndarray, bid_s: np.ndarray,
              ask_p: np.ndarray, ask_s: np.ndarray) -> np.ndarray:
    """Best-level order-flow imbalance of Cont, Kukanov & Stoikov (2014).

    e_n =  1{P_b(n) >= P_b(n-1)} q_b(n) - 1{P_b(n) <= P_b(n-1)} q_b(n-1)
         - 1{P_a(n) <= P_a(n-1)} q_a(n) + 1{P_a(n) >= P_a(n-1)} q_a(n-1)

    First event contributes 0 (no previous state).
    """
    e = np.zeros(len(bid_p))
    bp, bs, ap, as_ = bid_p[1:], bid_s[1:], ask_p[1:], ask_s[1:]
    bp0, bs0, ap0, as0 = bid_p[:-1], bid_s[:-1], ask_p[:-1], ask_s[:-1]
    e[1:] = ((bp >= bp0) * bs - (bp <= bp0) * bs0
             - (ap <= ap0) * as_ + (ap >= ap0) * as0)
    return e


def queue_imbalance(bid_s: np.ndarray, ask_s: np.ndarray) -> np.ndarray:
    """(q_b - q_a) / (q_b + q_a) at the touch, in [-1, 1]."""
    tot = bid_s + ask_s
    out = np.zeros(len(bid_s))
    nz = tot > 0
    out[nz] = (bid_s[nz] - ask_s[nz]) / tot[nz]
    return out


def depth_imbalance(book: pd.DataFrame, levels: int) -> np.ndarray:
    """Multi-level size imbalance over the top `levels` levels, in [-1, 1]."""
    bid = sum(book[f"bid_s{i}"].to_numpy() for i in range(1, levels + 1))
    ask = sum(book[f"ask_s{i}"].to_numpy() for i in range(1, levels + 1))
    return queue_imbalance(bid, ask)


def signed_trade_flow(msg_type: np.ndarray, msg_size: np.ndarray,
                      msg_direction: np.ndarray) -> np.ndarray:
    """Signed executed volume per event; + = buyer-initiated.

    LOBSTER direction on executions is the resting order's side, so the
    aggressor sign is -direction.
    """
    is_exec = np.isin(msg_type, (4, 5))
    return np.where(is_exec, -msg_direction * msg_size, 0.0)


def build_dataset(book: pd.DataFrame,
                  bin_seconds: float = 1.0,
                  horizons=(1, 2, 5, 10, 30, 60),
                  depth_levels: int = 5,
                  trim_seconds: float = 300.0) -> pd.DataFrame:
    """Aggregate event stream into a time-binned feature/target table.

    Flow features (OFI, trade flow) are summed within each bin; state features
    (queue imbalance, depth imbalance, spread) are the last value in the bin.
    Target ret_{h} is the mid-price log-return (in bps) from the end of bin t
    to the end of bin t+h. The first/last `trim_seconds` of the session are
    dropped (auction effects), as are the tail bins lacking a full horizon.
    """
    t = book["time"].to_numpy()
    bid_p = book["bid_p1"].to_numpy()
    bid_s = book["bid_s1"].to_numpy()
    ask_p = book["ask_p1"].to_numpy()
    ask_s = book["ask_s1"].to_numpy()

    mid = 0.5 * (bid_p + ask_p)
    spread = ask_p - bid_p

    ev = pd.DataFrame({
        "time": t,
        "ofi": event_ofi(bid_p, bid_s, ask_p, ask_s),
        "tflow": signed_trade_flow(book["msg_type"].to_numpy(),
                                   book["msg_size"].to_numpy(),
                                   book["msg_direction"].to_numpy()),
        "qi": queue_imbalance(bid_s, ask_s),
        "depth_imb": depth_imbalance(book, depth_levels),
        "mid": mid,
        "spread": spread,
    })

    start, end = t.min() + trim_seconds, t.max() - trim_seconds
    ev = ev[(ev["time"] >= start) & (ev["time"] <= end)]

    # right-closed bins: bin label = right edge; features known at that edge
    edges = np.arange(start, end + bin_seconds, bin_seconds)
    ev = ev.assign(bin=pd.cut(ev["time"], edges, labels=edges[1:], right=True))
    ev = ev.dropna(subset=["bin"])

    flows = ev.groupby("bin", observed=True)[["ofi", "tflow"]].sum()
    states = ev.groupby("bin", observed=True)[["qi", "depth_imb", "mid", "spread"]].last()
    df = flows.join(states)
    df.index = df.index.astype(float)

    # forward-fill state through empty bins; flows are genuinely zero there
    df = df.reindex(edges[1:])
    df[["ofi", "tflow"]] = df[["ofi", "tflow"]].fillna(0.0)
    df[["qi", "depth_imb", "mid", "spread"]] = df[["qi", "depth_imb", "mid", "spread"]].ffill()
    df = df.dropna(subset=["mid"])

    steps_per_h = int(round(1.0 / bin_seconds))
    for h in horizons:
        k = h * steps_per_h
        df[f"ret_{h}s"] = 1e4 * (np.log(df["mid"].shift(-k)) - np.log(df["mid"]))
    df = df.dropna()
    return df
