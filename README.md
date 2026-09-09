# Order book imbalance and short-horizon price moves

Tests whether simple order-book imbalance features predict mid-price moves a few
seconds out. Data is the free LOBSTER sample day (NASDAQ, 2012-06-21), level 10,
for MSFT, INTC and AMZN — about 1.56M order-book events across the three.

## Features

Four per-event features, aggregated onto a 1-second grid. Flow features are
summed within each bin, state features taken at the bin's end, and the first and
last 5 minutes of the session are dropped:

- `ofi` — best-level order-flow imbalance, following Cont, Kukanov and Stoikov (2014)
- `tflow` — signed executed volume. LOBSTER marks executions with the resting
  order's side, so the aggressor is the opposite sign
- `qi` — queue imbalance at the touch, (q_b − q_a)/(q_b + q_a)
- `depth_imb` — the same ratio over the top 5 levels

Loaders for the message/orderbook file pairs are in `src/lob/data.py` (LOBSTER
prices come as integers in units of 1e-4 USD).

Targets are forward mid-price log-returns in bps over 1/2/5/10/30/60 s, measured
bin edge to bin edge so they never overlap the feature window. Evaluation is 12
chronological folds with an expanding window: fit OLS on folds 1..i, test on
fold i+1, standardisation fit on the training data only. Per-fold and pooled R²
use the same zero-forecast benchmark. (An earlier version trained on one fold
at a time and mixed two R² benchmarks; the expanding window is more stable at
the long horizons and the tables below reflect it.)

## Results

Pooled out-of-sample R² of the full feature set, by horizon:

| horizon | MSFT | INTC | AMZN |
|---|---|---|---|
| 1s  | 3.0% | 1.8% | 1.3% |
| 2s  | 4.4% | 2.7% | 1.1% |
| 5s  | 6.0% | 4.5% | −0.1% |
| 10s | 5.0% | 4.5% | −0.4% |
| 30s | 1.1% | 1.0% | −1.5% |
| 60s | −3.2% | −4.0% | −1.9% |

Pooled numbers hide a lot of fold-to-fold noise on one day of data, so the
per-fold range matters: at 5s the fold R² runs −0.5% to +9.5% for MSFT, −3.6%
to +9.8% for INTC, −7.7% to +2.1% for AMZN (full ranges in
`results/metrics.csv`).

The signal peaks around 5s (R² ~5-6%, prediction-target correlation ~0.24-0.26)
in the large-tick names and is near zero by 30-60s. MSFT and INTC trade with
the spread pinned at one tick (~3-4 bps) and show it; small-tick AMZN (5.4 bps
median spread, thinner book) shows a little at 1-2s and nothing from 5s on.

Two things I didn't expect going in. OFI on its own doesn't predict at these
horizons (R² ~0) — the Cont et al. result is about contemporaneous impact, not
forecasting; the ablation in `metrics.csv` shows the difference. The
forward-looking part comes mostly from queue and depth imbalance.

And there's no tradeable edge: where R² is positive, the predicted move beats
half the spread in well under 1% of bins, so crossing the spread on this loses
money. Whether it survives inside a quoting or execution model is untested
here — that would need queue-position and fill modelling this repo doesn't do.

Per-fold ranges and the OFI-only ablation are in `results/metrics.csv`; decay
plot in `results/signal_decay.png`.

## Data

Not committed (~435MB). Grab the free LOBSTER sample files for MSFT, INTC and
AMZN, 2012-06-21, level 10 (`*_message_10.csv` + `*_orderbook_10.csv`) from
https://lobsterdata.com/info/DataSamples.php and put them in `./data`. There's a
mirror on Hugging Face (`totalorganfailure/lobster-data`) if the official links
are down.

One day, three tickers, from 2012. It's sample data, and I'm not claiming any of
this generalises across regimes. There's no queue-position or latency modelling
either; "tradeable" above just means beating the half-spread.

To run:

```
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_analysis.py
```
