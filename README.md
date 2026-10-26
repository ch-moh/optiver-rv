# Realized volatility forecasting from order-book data

Forecast the realized volatility of the **next 10 minutes** for ~100 stocks from
order-book snapshots, using the data of the Kaggle competition
[Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) (2021).
The competition is closed; this project evaluates models with local
cross-validation and the official metric (RMSPE), not on the leaderboard.

Kaggle notebook (same pipeline, with feature importance and plots):
[kaggle.com/code/chwalid/optiver-realized-volatility-lightgbm](https://www.kaggle.com/code/chwalid/optiver-realized-volatility-lightgbm)

## Approach
1. **Realized volatility** from the weighted average price (WAP):
   `RV = sqrt(sum r_t^2)`, with `r_t` the log-returns of the WAP.
2. **Naive baseline**: the realized volatility of the observed window, carried forward.
3. **Microstructure features**: RV at book levels 1 and 2, bid-ask spread,
   order-book imbalance, depth, number of updates, last-5-minute RV and its ratio
   to the full window, market-wide RV in the same window.
4. **Gradient boosting** (LightGBM) trained with weights `1/y²`, which makes the
   squared loss equivalent to RMSPE.
5. **Validation**: `GroupKFold` on `time_id`, so a time window is never split
   between training and validation (prevents leakage via the market-wide feature).

## Results
| Model | RMSPE (5-fold CV) |
|---|---|
| Naive (past RV) | 0.3414 |
| LightGBM | 0.2390 |

Relative improvement: 29.9725 %. 428,932 (stock_id, time_id) rows, LightGBM 4.7.0,
output of `python -m src.train --data data` (values printed to 4 decimals).
Features are built one stock partition at a time, so the full run fits in ~1 GB of RAM
(about 4 minutes on an 8 GB laptop).

## Run
```bash
pip install -r requirements.txt
kaggle competitions download -c optiver-realized-volatility-prediction -p data
unzip data/optiver-realized-volatility-prediction.zip -d data
python -m src.train --data data          # real data
python -m src.train --synthetic          # smoke test, no data needed
pytest -q
```

## Limitations and next steps
- Trade data (`trade_train.parquet`) is not used yet.
- `time_id`s are shuffled in the dataset, so validation is not chronological.
- Next: trade features, a linear HAR-style model as a second baseline,
  feature-importance analysis.
