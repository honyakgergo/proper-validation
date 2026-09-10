# The adapter Protocol

The contract that makes an audit reproducible, and the design decision that makes the whole tool
tractable. Generic re-execution of arbitrary research code is not achievable, and a tool that
breaks on a reviewer's notebook is a negative signal. A thin adapter is the alternative, and it is
a feature rather than an apology: it forces the strategy to be stated precisely enough to test.

## The whole contract, in one function

```python
import numpy as np

def strategy(data: np.ndarray) -> np.ndarray:
    """Return the position to hold, one row per row of `data`.

    `data` is a 2-D array of shape (T, n_features). Row t holds the
    information available at time t. The return value must have exactly T
    rows, and either:

      * shape (T,)     - one signal per row, for a single-instrument rule; or
      * shape (T, k)   - a book of k weights per row, for anything
                         cross-sectional.

    Either way row t is the position held *into* period t+1.
    """
```

Three requirements, all of them load-bearing:

1. **One row of output per row of input.** The behavioural leakage test compares signals before
   and after a cut point, and cannot align them otherwise.
2. **Deterministic.** Called several times on the same input, it must return the same thing. Seed
   anything stochastic inside the function.
3. **No hidden state.** No reading files, no module-level caches that survive between calls, no
   network. The function must be a pure map from the array it is given to the positions it returns.

A cross-sectional book is compared **per asset**, never reduced to one number per row. An
equal-weight top-3 rotation holds a gross exposure of 1.0 on nearly every row, so any scalar
summary of a row is blind to a leak that changes *which* names are held — which is the only kind
of leak a rotation can have. Measured on a nine-sector momentum rotation: a leak that moves 30
pre-cut books changes the largest individual weight by 0.33 and the gross exposure by exactly zero.

## You usually do not have to write this by hand

Most strategies are already a function of a labelled price frame. `qv.adapter.frame_adapter`
converts one into the callable above:

```python
from qv.adapter import frame_adapter

# momentum_positions(prices: pd.DataFrame, **params) -> pd.DataFrame
strategy = frame_adapter(
    momentum_positions, prices.index, prices.columns, lookback=252, skip=21, top_n=3
)
```

It closes over the index and the column labels and nothing else. A trading calendar and a universe
are both known in advance, so holding them fixed is not hindsight — but it does mean the test
clears the strategy's use of *prices* and says nothing about its use of the calendar. See the
blind spots below.

## What the leakage test does with it

```
signals_clean     = strategy(data)
signals_corrupted = strategy(data with everything from index C onward destroyed)
assert signals_clean[:C] == signals_corrupted[:C]
```

If any signal *before* the cut changes when only data *after* the cut is destroyed, the strategy
reads the future. This holds regardless of what the source code appears to say, which is why it
catches leaks buried inside third-party library calls that no static scan can reach.

Four corruption modes are used — shuffle, noise, constant, reverse — each repeated several times.
A signal keyed on the sign of a return has a coin flip's chance of surviving any single random
corruption unchanged, so one draw is not enough.

**Several cut points are swept, not one.** A leak can only surface if the signal actually moves
between its last revision before the cut and the cut itself, so cuts are placed one row after the
signal changes. A single fixed cut is close to blind on anything that rebalances less often than
every period: measured on the sector rotation with a known nine-session look-ahead injected, one
cut at 70% of the sample detected it *never*, twelve evenly spaced cuts caught it 5 times in 12,
and cuts placed after signal changes caught it 11 times out of 11.

The distance from the cut back to the earliest changed signal is the **look-ahead horizon**, and it
usually identifies the offending line immediately: a horizon of 1 is a `shift(-1)`, a horizon of 20
is a centred 40-period window, a horizon of hundreds is a full-sample statistic.

## What a clean result is worth

A clean result is reported with a **detection floor**, because it is easy to over-read.

A strategy that only revises its position monthly cannot betray a leak shorter than the gap
between revisions: the offending signal is overwritten at the next rebalance before anything
downstream sees it. So the report says *"no dependence on data more than about 22 periods after
the decision point"*, not *"no leakage"*. Two other things the test cannot settle:

- **A same-day decision.** The contract above defines row `t` as the position held *into* `t+1`,
  so using data at `t` is legitimate and produces no pre-cut change. Whether a strategy may act on
  the close it just observed is an accounting question about positions, not a look-ahead question.
- **The trading calendar.** The adapter supplies the index, so a rule keyed on "the last session of
  the month" is invariant to corruption. `qv scan` will flag the `shift(-1)` such a rule needs as
  `LEAK-NEGATIVE-SHIFT`, and that is a true positive by pattern and a false positive in substance —
  exactly the division of labour between the two tests.

## Writing one from a notebook

Find the cell that turns data into a signal. Lift it into a function. Replace anything that
references a global with a column of `data`.

```python
# In the notebook
df['ma_fast'] = df['close'].rolling(20).mean()
df['ma_slow'] = df['close'].rolling(100).mean()
df['signal']  = (df['ma_fast'] > df['ma_slow']).astype(float).shift(1)

# As an adapter. Column 0 is close.
def strategy(data: np.ndarray) -> np.ndarray:
    close = pd.Series(data[:, 0])
    fast = close.rolling(20).mean()
    slow = close.rolling(100).mean()
    signal = (fast > slow).astype(float).shift(1)
    return signal.fillna(0.0).to_numpy()
```

Keep the `.shift(1)`. Rewriting it away while porting is how an honest strategy becomes a leaky
one, and the perturbation test will catch you.

## Common mistakes

| Mistake | What happens |
|---|---|
| Returning `T - 1` rows after a `dropna()` | Rejected: the engine cannot align signals to rows |
| Output width that changes with the data | Rejected: it cannot be compared against its own baseline |
| Fitting a scaler on all of `data` inside the adapter | The behavioural test flags it, correctly - it is a real leak |
| Reading the original DataFrame from an enclosing scope | The corruption never reaches the strategy, so the test silently passes |
| Using `np.random` without a seed | Rejected as non-deterministic |

The fourth is the dangerous one. If the adapter closes over the original DataFrame instead of using
its `data` argument, corrupting `data` changes nothing and the test reports a clean bill of health
it has not earned. **Use only the array you are given** — or use `frame_adapter`, which has no
parameter that accepts values and so cannot express the mistake.
