# The interview protocol

Notebook forensics recover a lower bound on the trial count. This questionnaire is how the bound
gets revised upward, and how unquantifiable process bias becomes a manifest field.

Each answer either raises `n_trials`, populates a manifest field, or **disables a test the data
cannot support**. That last outcome is a legitimate result, not a failure: a test that cannot be
run honestly should not be run.

## Search process

1. **How many variants did you try before this one?** Include the ones you deleted. The notebook
   remembers what was saved; it cannot remember what was abandoned on a Tuesday three months ago.
2. **Did you change the date range after seeing results?** Each date range tried is a trial.
3. **Did you change the universe after seeing results?** Same.
4. **Did you look at out-of-sample performance before finalising parameters?** If yes, there is no
   out-of-sample period, and any test that treats one as held out must be disabled.
5. **Did you re-run the whole notebook with different settings?** How many times?

## Data provenance

6. **Where did the universe list come from, and when was it constructed?** If the universe came from an index,
    set `data.membership: sp500` or `nasdaq100` and the audit counts the missing names rather than
    accepting a declaration - the list is fetched and cached, nothing to download by hand. For any
    other index, point `data.membership_frame` at their own file. Do **not** accept a table of current members with a
    "date added" column: it reaches back decades and looks like history, but it contains only the
    survivors, and the audit refuses it for that reason. A list of instruments
   that exist today excludes everything that failed. Record the answer as
   `data.universe_point_in_time` in the manifest and pass it to `AuditInputs`: `false` raises
   `DATA-SURVIVORSHIP` and makes the result an upper bound, and leaving it unanswered puts the
   question in the report rather than out of sight.
6b. **Did the backtest credit uninvested cash?** A book that is 65% invested on average and pays
   nothing on the remainder understates its own return by the idle weight times the bill rate. The
   report quantifies this whenever positions are supplied; the researcher has to say whether it
   applies.
7. **Is the fundamental data point-in-time, or as-restated?** As-restated data contains revisions
   published months after the date they are attached to.
8. **How were missing values handled?** Backward fill is future data wearing the costume of data
   hygiene.

## Costs and implementation

9. **Where did the cost assumption come from?** If it is a round number chosen after the fact,
   record `COST-ASSUMED-NOT-DERIVED` and let break-even cost speak instead.
10. **At what size was this intended to trade?** Not modelled by this tool, but it belongs in the
    manifest so a reader can judge.
11. **Are the positions achievable?** Anything requiring the close of the same bar that generated
    the signal is not.

## Construction

12. **Was any scaler, imputer or encoder fitted before the train/test split?** The most common
    subtle leak, and the hardest to see in a notebook.
13. **Does any feature use a full-sample statistic?** A threshold set from the whole series encodes
    knowledge of the whole series.
14. **Is the target constructed with a negative shift?** Legitimate for a label, fatal for a
    feature. Confirm which.

## Closing question

15. **If this stopped working tomorrow, what would be the most likely reason?**

Ask it. The answer is frequently the finding - researchers usually know their strategy's weakest
joint, and are rarely asked to name it.
