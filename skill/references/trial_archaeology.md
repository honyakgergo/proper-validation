# Trial archaeology

Recovering how many configurations were really examined. `n_trials` is the key input to every
multiple-testing correction and the most under-reported number in backtesting - not usually through
dishonesty but because nobody counts.

A notebook remembers what its author does not.

```bash
qv trials research.ipynb --reported 1
qv trials research.ipynb --json
```

## What the evidence is

**Execution counts.** A notebook of 20 code cells whose highest execution count is 140 was run
through at least seven times. Re-runs of an analysis cell are how a parameter search leaves
fingerprints.

**Out-of-order execution.** Cells whose count is lower than an earlier cell prove the notebook was
not run top to bottom, so the saved output corresponds to no single clean run - worth knowing
independently of the trial count.

**Parameter literals.** Assigning `window = 20` in one cell and `window = 50` in another is two
trials, recorded in the file. Distinct values across parameter-looking names multiply.

**Explicit grids.** `itertools.product`, `ParameterGrid`, `GridSearchCV` - the search space stated
outright. The size of the grid is a floor on the trials run.

**Filename lineage.** A directory holding `model.ipynb`, `model_v2.ipynb` and
`model_final_fixed.ipynb` is showing three attempts that the writeup will describe as one.

## Why sources are combined by maximum, not product

They overlap heavily. A grid search also inflates the execution count, and re-running it also
leaves parameter literals behind. Multiplying them would manufacture a number the evidence does not
support, and a bound you cannot defend is worse than a smaller one you can - it hands the
researcher a reason to dismiss the whole audit.

Taking the maximum keeps the number defensible in an argument, which is the only place it matters.

## It is always a lower bound

Configurations tried and deleted leave no trace. Ideas abandoned before being typed leave less.
Report the number as a floor, every time, and use the interview protocol to revise it upward.

A lower bound is still worth a great deal. It is almost always far above the "1" that the writeup
implies, and moving from 1 to 40 changes the Deflated Sharpe far more than moving from 40 to 400 -
the correction is logarithmic in the trial count. Getting the researcher from "one" to "some" is
most of the value.
