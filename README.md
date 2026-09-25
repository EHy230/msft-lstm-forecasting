# MSFT Stock Forecasting with an LSTM (PyTorch)

A PyTorch LSTM that forecasts Microsoft's (MSFT) next-day closing price, trained on 2010–2025 data and tested on
2026 data it has never seen. The model is checked against simple baselines, so the results show whether it
adds anything over a naive guess.

**[Open the notebook →](MSFT_LSTM_Stock_Forecasting.ipynb)**

## Approach

- **Data:** daily MSFT and QQQ (Nasdaq-100 ETF) prices from Yahoo Finance via `yfinance`, 2010 to present
- **Target:** tomorrow's percent return, converted back into a price for evaluation. Raw prices keep rising to
  levels the model never saw in training, and neural networks extrapolate poorly, so returns work better.
- **Features (50-day windows):** daily return, intraday range, open-to-close move, volume vs. 20-day average,
  distance from 10- and 50-day moving averages, 14-day RSI and the Nasdaq-100 daily return
- **Model:** two-layer LSTM (64 hidden units, dropout 0.2) followed by two fully connected layers, trained with L1 loss and Adam
- **Evaluation setup:** date-based split (train 2010–2024, validation 2025, test 2026), scalers fitted on training
  data only, and early stopping on validation loss

## Results (2026 test set, 182 trading days)

| Model | MAPE | Direction accuracy |
|---|---|---|
| Version 1: LSTM on raw prices | 2.43% | – |
| **Version 2: LSTM on returns + engineered features** | **1.58%** | **49.5%** |
| Baseline: tomorrow's price = today's price | 1.57% | – |
| Baseline: always guess "up" | – | 48.9% |

- Predicting returns instead of prices **cut the error by about a third** and removed version 1's systematic lag.
- The model **ties the naive baseline** and its direction accuracy is **about a coin flip**. Early stopping ended
  training at epoch 17 because validation loss stopped improving: the network found no pattern in past prices
  that carried over to new data.
- This matches the efficient-market view that daily returns are close to unpredictable from price history alone.
  The main takeaway is the evaluation method: without baselines and a strict date split, an LSTM price chart can
  look impressive while adding nothing over "tomorrow = today".

*Results are from a run on September 24, 2026. The notebook downloads data up to the day it's run, so the numbers
will change slightly if you rerun it.*

## Run it yourself

The easiest way is [Google Colab](https://colab.research.google.com/): **File → Upload notebook**, then **Runtime → Run all**.

To run it locally:

```bash
pip install -r requirements.txt
jupyter notebook MSFT_LSTM_Stock_Forecasting.ipynb
```

