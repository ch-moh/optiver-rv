import numpy as np
import pandas as pd

from src.features import wap, realized_volatility, book_features
from src.train import rmspe


def _book(prices):
    n = len(prices)
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({
        "stock_id": 0, "time_id": 0, "seconds_in_bucket": np.arange(n) * 10,
        "bid_price1": p - 0.01, "ask_price1": p + 0.01,
        "bid_price2": p - 0.02, "ask_price2": p + 0.02,
        "bid_size1": 100, "ask_size1": 100, "bid_size2": 100, "ask_size2": 100,
    })


def test_wap_equal_sizes_is_mid():
    df = _book([1.0, 1.1])
    assert np.allclose(wap(df), [1.0, 1.1])


def test_constant_price_has_zero_volatility():
    feats = book_features(_book([1.0] * 10))
    assert feats.loc[0, "rv1"] == 0.0


def test_realized_volatility_matches_definition():
    r = pd.Series([np.nan, 0.01, -0.02, 0.005])
    assert np.isclose(realized_volatility(r), np.sqrt(0.01**2 + 0.02**2 + 0.005**2))


def test_rmspe_perfect_and_scale():
    y = np.array([0.001, 0.002])
    assert rmspe(y, y) == 0.0
    assert np.isclose(rmspe(y, 1.1 * y), 0.1)
