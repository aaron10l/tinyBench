# Benchmark Question Templates + Phenomena Injections

## 1) Which group is riskier? - DONE

### Question template
Between groups **{GROUP_A}** and **{GROUP_B}** (defined by **{GROUP_COL}**), which group should be treated as higher risk with respect to **{RISK_METRIC_COL}**?  
Return **exactly one** group name.

### Corresponding phenomena injection
Make group means similar but variances different:
- Same mean of **{RISK_METRIC_COL}** across groups
- Much higher variance (or heavier tails / outliers) for one group

### Example (original)
**Which group is riskier?**

- Phenomenon: Heteroskedastic noise with identical means but different variances  
- Corresponding injector script: `src/injectors/heteroskedastic_injection.py`


## 2) Find the strongest data quality filter - DONE

### Question template
You can apply **exactly one** rule based on a **single column** to exclude problematic rows before modeling **{OUTCOME_COL}**.  
Which column should the rule be based on? Return a **single column name**.

### Corresponding phenomena injection
Inject a single feature that implicitly identifies rows with corrupted or nonsensical outcomes (e.g., a sensor/ETL flag, sentinel-coded values, or a measurement that becomes impossible when the pipeline breaks). Ensure removing rows selected by a simple condition on that feature dramatically improves consistency of **{OUTCOME_COL}** patterns.

### Example (original)
You can apply exactly one rule based on a single column to exclude problematic rows before modeling outcome. Which column should the rule be based on? Return a single column name.

- Phenomenon: A hidden “bad rows” indicator column correlates with corrupted labels.  
- Corresponding injector script: `src/injectors/bad_rows_injection.py`


## 3) Identify important features for the outcome (robust to misleading names) - DONE

### Question template
Which columns matter most for understanding or predicting **{OUTCOME_COL}**?  
Return up to **K** feature names, ranked from most to least important, with a brief justification grounded in patterns in the table.

### Corresponding phenomena injection
One (or both) of:
- **Leakage injection:** add a feature that is implausibly predictive because it encodes target or future information (directly or via a proxy).
- **Name swap / misleading schema:** swap column names (or rename) so naive heuristics based on names fail, but statistical dependence still reveals the true drivers.

### Example (original)
**Which features are important to understand the outcome?**

- Phenomenon: Data leakage on “likely” feature names; swap column names  
- Corresponding injector script: `src/injectors/name_swap_injection.py`


## 4) Did performance improve?

### Question template
Between period A and period B, did **{OUTCOME_COL}** improve, worsen, or stay about the same?  
Return **exactly one** of: **IMPROVED**, **WORSENED**, **UNCHANGED**.

### Corresponding phenomena injection
Create a composition shift: change group sampling rates (or mixture weights) across periods so that:
- Within each group, performance improves (or worsens), but
- The overall aggregate shows the opposite

### Example (original)
**Did performance improve from last quarter to this quarter?**

- Phenomenon: Change sampling rate across quarters → Simpson’s paradox  
- Corresponding injector script: `src/injectors/simpsons_paradox_injection.py`


## 5) Detect when to retrain

### Question template
Column **{PRED_COL}** is a model’s output score over the rows in this table.  
Should the model be retrained? Return **YES** or **NO**.  
If **YES**, return the earliest row (or timestamp) after which retraining is warranted.

### Corresponding phenomena injection
Introduce a change-point in the data-generating relationship while keeping marginals stable:
- After row **t**, set **{TARGET_OR_PRED_COL}** to depend on an interaction (e.g., XOR-like, piecewise, multiplicative) of two features
- Ensure the mean of **{PRED_COL}** (or target) stays roughly constant

### Example (original)
Column X is the output of my predictive model. Should we retrain it? If yes, at what row should we retrain it?

- Phenomenon: Introduce interaction effect halfway through, keep average of X unchanged  
- Corresponding injector script: `src/injectors/changepoint_injection.py`