"""Train and evaluate: naive baseline vs gradient boosting, metric RMSPE.

Usage:
    python -m src.train --data data/            # real Kaggle data
    python -m src.train --synthetic             # smoke test without data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features import book_features, add_cross_sectional

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # fallback so the pipeline still runs
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGB = False


def rmspe(y_true, y_pred) -> float:
    """Root mean squared percentage error, the official competition metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def load_real(data_dir: Path) -> pd.DataFrame:
    # The full book (~167M rows) does not fit in memory: build features one stock
    # partition at a time, then add the cross-sectional feature on the small table.
    parts = sorted((data_dir / "book_train.parquet").glob("stock_id=*"),
                   key=lambda p: int(p.name.split("=")[1]))
    feats = []
    for i, part in enumerate(parts, 1):
        book = pd.read_parquet(part)
        book["stock_id"] = int(part.name.split("=")[1])
        feats.append(book_features(book))
        print(f"features {i}/{len(parts)} ({part.name})", flush=True)
    feats = add_cross_sectional(pd.concat(feats, ignore_index=True))
    train = pd.read_csv(data_dir / "train.csv")
    return train.merge(feats, on=["stock_id", "time_id"], how="inner")


def make_model(seed: int):
    if HAS_LGB:
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 random_state=seed, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)


def evaluate(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    feature_cols = [c for c in df.columns if c not in ("stock_id", "time_id", "target", "row_id")]
    X, y, groups = df[feature_cols], df["target"].values, df["time_id"].values
    # RMSPE is equivalent to weighted MSE with weights 1/y^2
    w = 1.0 / np.square(y)

    oof = np.zeros(len(df))
    # Group by time_id: the same 10-min window never appears in both train and validation,
    # so the cross-sectional market feature cannot leak the answer.
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model(seed)
        model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        oof[va] = model.predict(X.iloc[va])
    oof = np.clip(oof, 1e-6, None)

    naive = df["rv1"].values  # past 10-min realized vol carried forward
    return {
        "n_rows": len(df),
        "model": "LightGBM" if HAS_LGB else "HistGradientBoosting (fallback)",
        "rmspe_naive": rmspe(y, naive),
        "rmspe_model": rmspe(y, oof),
        "improvement_pct": 100 * (1 - rmspe(y, oof) / rmspe(y, naive)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    if args.synthetic:
        from scripts.make_synthetic import make_dataset
        book, train = make_dataset()
        df = train.merge(add_cross_sectional(book_features(book)), on=["stock_id", "time_id"])
    else:
        df = load_real(args.data)
    res = evaluate(df)
    for k, v in res.items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
