import numpy as np
import pandas as pd
import pytest

from lob.features import (event_ofi, queue_imbalance, signed_trade_flow,
                          build_dataset)


def test_ofi_hand_computed_sequence():
    # state 0: bid 100@10, ask 101@8
    # state 1: bid price up to 100.5@4          -> e = +4 (bid improves: +q_b(n))
    #          ask unchanged 101@8              -> ask terms: -8 + 8 = 0
    # state 2: bid unchanged, ask size drops to 5 at same price
    #          bid terms: p equal -> +3 - 4 = wait, bid unchanged: +4 - 4 = 0
    #          ask terms: p equal -> -5 + 8 = +3
    # state 3: bid drops to 100@10, ask unchanged
    #          bid terms: p down -> -4 ; ask terms: 0
    bid_p = np.array([100.0, 100.5, 100.5, 100.0])
    bid_s = np.array([10.0, 4.0, 4.0, 10.0])
    ask_p = np.array([101.0, 101.0, 101.0, 101.0])
    ask_s = np.array([8.0, 8.0, 5.0, 5.0])
    e = event_ofi(bid_p, bid_s, ask_p, ask_s)
    assert e[0] == 0.0
    assert e[1] == pytest.approx(4.0)
    assert e[2] == pytest.approx(3.0)
    assert e[3] == pytest.approx(-4.0)


def test_queue_imbalance_bounds_and_signs():
    qi = queue_imbalance(np.array([10.0, 0.0, 5.0]), np.array([0.0, 10.0, 5.0]))
    assert np.allclose(qi, [1.0, -1.0, 0.0])
    rng = np.random.default_rng(0)
    qi = queue_imbalance(rng.uniform(0, 100, 1000), rng.uniform(0, 100, 1000))
    assert np.all(qi <= 1.0) and np.all(qi >= -1.0)


def test_signed_trade_flow_aggressor_sign():
    # execution against a resting SELL order (direction -1) is buyer-initiated -> positive
    types = np.array([4, 4, 1, 3])
    sizes = np.array([100.0, 50.0, 200.0, 10.0])
    dirs = np.array([-1, 1, 1, -1])
    tf = signed_trade_flow(types, sizes, dirs)
    assert np.allclose(tf, [100.0, -50.0, 0.0, 0.0])


def _toy_book(n_events: int = 4000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(34200, 36000, n_events))
    mid = 100 + np.cumsum(rng.normal(0, 0.001, n_events))
    spread = 0.01
    df = pd.DataFrame({
        "time": t,
        "bid_p1": mid - spread / 2, "ask_p1": mid + spread / 2,
        "bid_s1": rng.integers(1, 500, n_events).astype(float),
        "ask_s1": rng.integers(1, 500, n_events).astype(float),
        "msg_type": rng.choice([1, 2, 3, 4], n_events),
        "msg_size": rng.integers(1, 100, n_events).astype(float),
        "msg_direction": rng.choice([-1, 1], n_events),
    })
    for i in range(2, 6):
        df[f"bid_p{i}"] = df["bid_p1"] - (i - 1) * spread
        df[f"ask_p{i}"] = df["ask_p1"] + (i - 1) * spread
        df[f"bid_s{i}"] = rng.integers(1, 500, n_events).astype(float)
        df[f"ask_s{i}"] = rng.integers(1, 500, n_events).astype(float)
    return df


def test_no_lookahead_in_features():
    """Features for early bins must be identical whether or not later events exist."""
    book = _toy_book()
    full = build_dataset(book, bin_seconds=1.0, horizons=(1,), depth_levels=5,
                         trim_seconds=60.0)
    cutoff_time = book["time"].quantile(0.7)
    truncated = build_dataset(book[book["time"] <= cutoff_time], bin_seconds=1.0,
                              horizons=(1,), depth_levels=5, trim_seconds=60.0)
    feature_cols = ["ofi", "tflow", "qi", "depth_imb", "spread"]
    common = full.index.intersection(truncated.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(full.loc[common, feature_cols],
                                  truncated.loc[common, feature_cols])


def test_target_is_strictly_forward_looking():
    """ret_h at bin t must equal the mid log-return from bin edge t to t+h."""
    book = _toy_book()
    df = build_dataset(book, bin_seconds=1.0, horizons=(5,), depth_levels=5,
                       trim_seconds=60.0)
    t0 = df.index[10]
    expected = 1e4 * (np.log(df["mid"].loc[t0 + 5]) - np.log(df["mid"].loc[t0]))
    assert df["ret_5s"].loc[t0] == pytest.approx(expected)
