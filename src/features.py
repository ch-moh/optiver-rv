"""Order-book features for realized volatility forecasting (Optiver data format).

Each (stock_id, time_id) is a 10-minute bucket of order-book snapshots.
The target is the realized volatility of the NEXT 10 minutes.
"""
import numpy as np
import pandas as pd


def wap(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """Weighted average price: bid weighted by ask size and vice versa."""
    b, a = df[f"bid_price{level}"], df[f"ask_price{level}"]
    bs, as_ = df[f"bid_size{level}"], df[f"ask_size{level}"]
    return (b * as_ + a * bs) / (bs + as_)


def log_return(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def realized_volatility(returns: pd.Series) -> float:
    """sqrt(sum r_t^2) — ignores NaN (first return of each bucket)."""
    return float(np.sqrt(np.nansum(np.square(returns))))


def book_features(book: pd.DataFrame) -> pd.DataFrame:
    """Aggregate snapshot-level book data into one row per (stock_id, time_id).

    Expected columns: stock_id, time_id, seconds_in_bucket,
    bid_price1/2, ask_price1/2, bid_size1/2, ask_size1/2.
    """
    df = book.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).copy()
    df["wap1"] = wap(df, 1)
    df["wap2"] = wap(df, 2)
    g = df.groupby(["stock_id", "time_id"], sort=False)
    df["ret1"] = g["wap1"].transform(log_return)
    df["ret2"] = g["wap2"].transform(log_return)
    df["spread"] = df["ask_price1"] / df["bid_price1"] - 1
    df["imbalance"] = (df["bid_size1"] - df["ask_size1"]).abs() / (df["bid_size1"] + df["ask_size1"])
    df["depth"] = df[["bid_size1", "bid_size2", "ask_size1", "ask_size2"]].sum(axis=1)

    def agg(sub: pd.DataFrame, suffix: str) -> pd.DataFrame:
        gg = sub.groupby(["stock_id", "time_id"])
        out = pd.DataFrame({
            f"rv1{suffix}": gg["ret1"].apply(realized_volatility),
            f"rv2{suffix}": gg["ret2"].apply(realized_volatility),
            f"spread_mean{suffix}": gg["spread"].mean(),
            f"imbalance_mean{suffix}": gg["imbalance"].mean(),
            f"depth_mean{suffix}": gg["depth"].mean(),
            f"n_updates{suffix}": gg["seconds_in_bucket"].count(),
        })
        return out

    full = agg(df, "")
    last_half = agg(df[df["seconds_in_bucket"] >= 300], "_last300")
    feats = full.join(last_half, how="left").reset_index()
    # Ratio of recent to full-window volatility: captures intra-bucket regime change
    feats["rv1_ratio_last300"] = feats["rv1_last300"] / feats["rv1"].replace(0, np.nan)
    return feats


def add_cross_sectional(feats: pd.DataFrame) -> pd.DataFrame:
    """Market-wide volatility in the same time_id (mean across stocks)."""
    out = feats.copy()
    out["rv1_market_mean"] = out.groupby("time_id")["rv1"].transform("mean")
    out["rv1_rel_market"] = out["rv1"] / out["rv1_market_mean"].replace(0, np.nan)
    return out
