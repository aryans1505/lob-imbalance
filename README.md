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
chronological folds: fit OLS on fold i, test on fold i+1, standardisation fit on
the training fold only. R² is pooled over all test folds against a zero forecast.

## Results

Pooled out-of-sample R² of the full feature set, by horizon:

| horizon | MSFT | INTC | AMZN |
|---|---|---|---|
| 1s  | 2.3% | 1.6% | −0.7% |
| 2s  | 3.4% | 2.7% | −0.7% |
| 5s  | 4.1% | 3.9% | −2.2% |
| 10s | 1.8% | 3.1% | −3.5% |
| 30s | −4.6% | −4.2% | −6.9% |
| 60s | −13.2% | −13.8% | −10.0% |

The signal peaks around 5s (R² ~4%, prediction-target correlation ~0.23) and is
gone by 30s. It's a large-tick thing: MSFT and INTC trade with the spread pinned
at one tick (~3-4 bps) and show it, while AMZN (5.4 bps median spread, thinner
book) shows nothing under the same model.

Two things I didn't expect going in. OFI on its own doesn't predict at these
horizons (R² ~0) — the Cont et al. result is about contemporaneous impact, not
forecasting; the ablation in `metrics.csv` shows the difference. The
forward-looking part comes mostly from queue and depth imbalance.

And there's no tradeable edge: where R² is positive, the predicted move beats
half the spread in under 1% of bins. Crossing the spread on this loses money.
If it's useful anywhere it's in quoting or execution timing.

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
