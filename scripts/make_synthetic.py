"""Synthetic order-book data in the Optiver format, for smoke tests only.

Volatility is persistent within a stock and shares a market factor across stocks,
so the next-window volatility is partly predictable from the current window.
This is NOT a substitute for the real data: results on it mean nothing.
"""
import numpy as np
import pandas as pd


def make_dataset(n_stocks: int = 8, n_times: int = 120, n_snap: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    market = np.exp(rng.normal(0, 0.35, n_times))
    rows, targets = [], []
    for s in range(n_stocks):
        base = rng.uniform(0.0008, 0.002)
        for t in range(n_times):
            sigma = base * market[t] * np.exp(rng.normal(0, 0.15))
            secs = np.sort(rng.choice(600, n_snap, replace=False))
            mid = 1.0 * np.exp(np.cumsum(rng.normal(0, sigma / np.sqrt(n_snap), n_snap)))
            half = mid * rng.uniform(1e-4, 4e-4, n_snap)
            bs1, as1 = rng.integers(1, 500, n_snap), rng.integers(1, 500, n_snap)
            rows.append(pd.DataFrame({
                "stock_id": s, "time_id": t, "seconds_in_bucket": secs,
                "bid_price1": mid - half, "ask_price1": mid + half,
                "bid_price2": mid - 2 * half, "ask_price2": mid + 2 * half,
                "bid_size1": bs1, "ask_size1": as1,
                "bid_size2": rng.integers(1, 500, n_snap), "ask_size2": rng.integers(1, 500, n_snap),
            }))
            targets.append((s, t, sigma * np.exp(rng.normal(0, 0.2))))
    book = pd.concat(rows, ignore_index=True)
    train = pd.DataFrame(targets, columns=["stock_id", "time_id", "target"])
    return book, train
