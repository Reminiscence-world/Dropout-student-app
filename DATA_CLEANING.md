# Data Cleaning Policy — School Dropout Dataset (`data/SIH-Dataset.csv`)

This document records two known defects found in the raw school dataset,
the exact rule applied to each, and why. The original CSV file is never
modified — cleaning is applied in memory, deterministically, inside
`src/school_pipeline.py` (`clean_school_data`), every time the model is
trained.

**These decisions were made strictly for data integrity and
reproducibility. They were not chosen to raise or lower model
performance**, and were finalized before the cleaned model was ever
trained or evaluated.

---

## 1. `Teaching_Staff` — 84 rows with an invalid value

**What was found:** `Teaching_Staff` should only ever contain `Good`,
`Poor`, or `Excellent`. 84 of 10,198 rows (0.82%) instead contain `Male`
or `Female` — values that belong to the `Gender` column, not this one.

**Why this is treated as a defect, not real data:** these 84 rows do not
simply duplicate that student's own `Gender` value (only 64 of the 84
match; 20 are mismatched — e.g. a `Female` student with
`Teaching_Staff="Male"`). No row elsewhere in the dataset shares the same
full profile (School type, Location, Infrastructure, Gender, Caste, Age,
Standard, Socioeconomic status, and outcome) with a valid
`Teaching_Staff` value, so the correct value cannot be recovered from the
data itself. The 84 rows also aren't scattered randomly — they fall in
tight repeating position blocks at six locations in the file — consistent
with a bug in whatever process generated or corrupted this slice of rows,
not scattered manual data-entry mistakes.

**Rule applied:** any `Teaching_Staff` value outside `{Good, Poor,
Excellent}` is recoded to an explicit `"Unknown"` category. The row is
**kept** — the other 8 feature columns for these students are legitimate
and are not discarded.

**Why recode instead of drop or guess:** guessing a value (e.g. the
column's mode) would fabricate data with no evidential basis, which is a
worse integrity violation than leaving the gap visible. Dropping the rows
would discard otherwise-valid data for 84 students, and would remove a
non-randomly-distributed slice of the dataset, risking an unknown
downstream bias. An explicit `Unknown` category keeps the data honest
about what isn't known.

**Scale:** 84 / 10,198 rows affected (0.82%).

---

## 2. `Location` — 2 rows with a missing value

**What was found:** 2 of 10,198 rows (0.02%) have a null `Location`.
Unlike the `Teaching_Staff` defect, these 2 rows are isolated (row
positions 962 and 6061 — far apart, no repeating pattern).

**Rule applied:** these 2 rows are **dropped**.

**Why drop instead of recode:** at 0.02% of the dataset, adding a
permanent new categorical level (and therefore a new one-hot-encoded
column that is non-zero for only 2 rows) for a genuinely negligible,
apparently isolated gap isn't warranted. Simple listwise deletion is
standard practice at this scale, fully deterministic, and has no
meaningful effect on class balance or any feature's distribution.

**Scale:** 2 / 10,198 rows affected (0.02%).

---

## Net effect on dataset size

| | Rows |
|---|---|
| Raw dataset | 10,198 |
| After cleaning (84 recoded, 2 dropped) | 10,196 |

## Where this is applied

`src/school_pipeline.py → clean_school_data(df)`, called by
`train_school_model.py` before validation and training. The function
operates on an in-memory copy — `data/SIH-Dataset.csv` on disk is never
modified. Re-running the training script reproduces an identical result
every time, from the same original CSV.
