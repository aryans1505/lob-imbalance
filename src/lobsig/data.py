"""Loaders for LOBSTER message/orderbook CSV pairs.

LOBSTER format (https://lobsterdata.com):
- message file columns: time (sec after midnight), type, order_id, size, price, direction
  * type: 1=new limit, 2=partial cancel, 3=delete, 4=execution (visible), 5=execution (hidden), 7=halt
  * direction: 1 = buy limit order, -1 = sell limit order. For executions (type 4/5) the
    direction is that of the RESTING limit order, so the aggressor side is -direction.
- orderbook file columns (level N): ask_p1, ask_s1, bid_p1, bid_s1, ..., ask_pN, ask_sN, bid_pN, bid_sN
- prices are integers in units of 1e-4 USD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SCALE = 1e-4

MESSAGE_COLS = ["time", "type", "order_id", "size", "price", "direction"]


def load_messages(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=MESSAGE_COLS)
    df["price"] = df["price"] * PRICE_SCALE
    return df


def load_orderbook(path: str, levels: int) -> pd.DataFrame:
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_p{i}", f"ask_s{i}", f"bid_p{i}", f"bid_s{i}"]
    df = pd.read_csv(path, header=None, names=cols, dtype=np.float64)
    for c in df.columns:
        if c.startswith(("ask_p", "bid_p")):
            df[c] = df[c] * PRICE_SCALE
    return df


def load_day(message_path: str, orderbook_path: str, levels: int = 10) -> pd.DataFrame:
    """Combine message timestamps with book states; one row per book-changing event."""
    msg = load_messages(message_path)
    book = load_orderbook(orderbook_path, levels)
    if len(msg) != len(book):
        raise ValueError(f"message rows ({len(msg)}) != orderbook rows ({len(book)})")
    book.insert(0, "time", msg["time"].to_numpy())
    book["msg_type"] = msg["type"].to_numpy()
    book["msg_size"] = msg["size"].to_numpy()
    book["msg_direction"] = msg["direction"].to_numpy()
    return book
