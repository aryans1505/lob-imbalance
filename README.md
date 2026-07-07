# Order-Book Imbalance and Short-Horizon Price Predictability

Do simple limit-order-book imbalance features predict short-horizon mid-price moves?
A small, honest research pipeline on real NASDAQ level-10 data (LOBSTER sample day,
2012-06-21): MSFT, INTC, AMZN — 1.56M order-book events.

## Method

- **Data**: LOBSTER message + orderbook files (level 10), one full trading session
  per ticker, prices in 1e-4 USD. Loaders in `src/lobsig/data.py`.
- **Features** (`src/lobsig/features.py`), aggregated on a right-closed 1-second grid,
  first/last 5 minutes trimmed:
  - `ofi` — best-level order-flow imbalance (Cont, Kukanov & Stoikov, 2014), summed per bin
  - `tflow` — signed executed volume (aggressor side = minus LOBSTER direction), summed
  - `qi` — queue imbalance `(q_b − q_a)/(q_b + q_a)` at the touch, end-of-bin state
  - `depth_imb` — same imbalance over the top 5 levels
- **Targets**: forward mid-price log-returns in bps over 1/2/5/10/30/60 s, from bin edge
  to bin edge — strictly out of the feature window.
- **Evaluation** (`src/lobsig/evaluate.py`): 12 contiguous chronological folds; train OLS
  on fold *i*, test on fold *i+1*; standardisation fit on train only. Pooled OOS R²
  against a zero-forecast benchmark. No shuffling anywhere.
- **Leakage controls**: unit tests assert features are unchanged when future events are
  deleted, and that targets are strictly forward-looking (`tests/test_features.py`).

## Results (walk-forward, out-of-sample)

Pooled OOS R² of the full feature set by horizon:

| horizon | MSFT | INTC | AMZN |
|---|---|---|---|
| 1s  | 2.3% | 1.6% | −0.7% |
| 2s  | 3.4% | 2.7% | −0.7% |
| 5s  | **4.1%** | **3.9%** | −2.2% |
| 10s | 1.8% | 3.1% | −3.5% |
| 30s | −4.6% | −4.2% | −6.9% |
| 60s | −13.2% | −13.8% | −10.0% |

(Full table incl. per-fold ranges, prediction-target correlations and OFI-only
ablation: `results/metrics.csv`; decay plot: `results/signal_decay.png`.)

## What the results actually say

1. **Predictability is real but tiny and short-lived**: it peaks around 5 s
   (OOS R² ≈ 4%, prediction–target correlation ≈ 0.23) and is gone by 30 s.
2. **It is a large-tick phenomenon**: MSFT and INTC (spread pinned at 1 tick,
   ~3–4 bps) show it; small-tick AMZN (5.4 bps median spread, sparser book) shows
   none under the same linear model. Consistent with the queue-imbalance literature.
3. **OFI alone does not forecast** at these horizons (OOS R² ≈ 0): Cont et al.'s
   result is about *contemporaneous* price impact, not prediction — the ablation
   in `metrics.csv` makes that distinction concrete. The forward-looking signal
   here comes mainly from queue/depth imbalance.
4. **No naive tradeable edge**: at the horizons where R² is positive, the
   predicted move exceeds half the concurrent spread in <1% of bins. Crossing
   the spread on this signal loses money; it is at best an input to passive
   execution/quoting decisions.

## Limitations (deliberately not hidden)

- One trading day, three tickers, from 2012 — a public sample, not a research
  dataset. Nothing here is claimed to generalise across regimes.
- Linear model on 1-second bins; event-time sampling and nonlinear models are
  natural extensions.
- No queue-position or latency modelling; "tradeable" is assessed only against
  the half-spread bound.

## Reproduce

```
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_analysis.py
```

Data (not committed — ~435MB): the free LOBSTER sample files for MSFT/INTC/AMZN,
2012-06-21, level 10 (`*_message_10.csv` + `*_orderbook_10.csv`) placed in `./data`.
Source: https://lobsterdata.com/info/DataSamples.php (also mirrored on Hugging Face
at `totalorganfailure/lobster-data` if the official links are down).
