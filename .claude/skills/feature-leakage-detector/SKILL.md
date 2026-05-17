---
name: feature-leakage-detector
description: Audits feature engineering and dataset splits for the five canonical leakage patterns — target leakage, temporal leakage, group/entity leakage, train-test contamination, and label leakage via target encoding. Use whenever code creates features, defines a CV split, joins external data, or computes group statistics. Senior interviewers probe for these — block any PR that fails the checks here.
---

# Feature Leakage Detector

## When this skill applies

Trigger this skill whenever code:
- Adds or transforms a feature in `src/**/features/` or any notebook fitting a model.
- Defines a train/val/test split, including any sklearn `KFold`, `GroupKFold`, `TimeSeriesSplit`, or custom split.
- Computes any group-level aggregate (mean target by category, rolling stats, target encoding).
- Joins external data (weather, holidays, FX rates, anything time-keyed).
- Calls `fit_transform` on a transformer that has internal state (Scaler, Encoder, Imputer, PCA).

If any of the above appears without the checks below being addressed, treat it as a failed task.

## The five leakage patterns — refuse code that hits any of them

### 1. Target leakage (feature is a function of the target itself, directly or via proxy)

**Pattern:** `total_amount` used as a feature when predicting `base_amount` (total includes the base). `actual_arrival_time` used to predict `delay`. `payment_completed_flag` used to predict `will_pay`.

**Test:**
```python
# If any single feature's correlation with the target exceeds 0.95, flag it.
high_corr = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
assert (high_corr > 0.95).sum() == 0, f"Likely target leak: {high_corr[high_corr > 0.95].to_dict()}"
```

Plus a manual review: for every feature, ask "could this value be known *strictly before* the target is realized in production?" If not, drop it.

### 2. Temporal leakage (training on future, evaluating on past)

**Patterns:**
- Random `train_test_split` on time-series data. Forbidden.
- Imputing missing values using the median of the *full* dataset before splitting.
- Computing rolling features (e.g. 7-day mean fare) on a window that crosses the split boundary.
- Joining a SCD2 dimension (e.g. driver tier) using the *current* version instead of the version valid at event time.

**Test:**
```python
# 1. Splits must be temporal:
assert train["pickup_datetime"].max() < val["pickup_datetime"].min(), "Temporal overlap"
assert val["pickup_datetime"].max()   < test["pickup_datetime"].min(), "Temporal overlap"

# 2. Any rolling/lag feature is computed within each row's history only:
#    Use df.groupby(key).rolling(window, closed='left'), NEVER 'both'.

# 3. Imputers/scalers must be fit on TRAIN ONLY:
scaler.fit(X_train)  # NOT fit(X_all)
X_val = scaler.transform(X_val)
```

### 3. Group / entity leakage (same entity in train and test)

**Pattern:** Same `driver_id` or `vehicle_id` appears in both train and test splits, so the model memorizes per-driver tipping habits instead of learning generalizable patterns.

**Test:**
```python
overlap = set(train["driver_id"]) & set(test["driver_id"])
# Decide deliberately: if you NEED per-entity generalization, this set should be empty (use GroupKFold).
# If per-entity is fine, document it and move on. Don't leave it undecided.
```

Decide based on cardinality and intent. Low-cardinality categoricals like region IDs, product categories, or store IDs typically *should* overlap between splits — they're shared label-space members, not entities to generalize across. High-cardinality identifiers (user_id, device_id, medallion_id) usually *should not* overlap if you want the model to generalize to unseen entities.

### 4. Train-test contamination via shared preprocessing state

**Patterns:**
- Fitting a `StandardScaler`, `OneHotEncoder(handle_unknown='ignore')`, `TargetEncoder`, or PCA on the union of train+test.
- Computing the vocab of a tokenizer over the whole dataset.
- Imputing nulls with the global median.
- Computing TF-IDF or word2vec embeddings on the whole corpus.

**Test:** Audit every fitted transformer. Each must be wrapped in a `Pipeline` and fit only on training folds. For CV, use `cross_val_score` with a pipeline so each fold refits.

### 5. Target encoding leakage

**Pattern:** Replacing a categorical with `df.groupby(cat)[target].mean()` computed on the full training set. The mean for category `c` includes the row's own target value → leak.

**Test:** Use out-of-fold target encoding (`category_encoders.TargetEncoder` with CV smoothing, or `sklearn.preprocessing.TargetEncoder` in 1.3+). Or use leave-one-out encoding. Never the naive groupby-mean.

```python
from sklearn.preprocessing import TargetEncoder
enc = TargetEncoder(smooth="auto", cv=5)  # cv arg is the leak protection
X_train_enc = enc.fit_transform(X_train, y_train)
X_test_enc  = enc.transform(X_test)
```

## The audit checklist — run before any model is logged to MLflow

Copy this into `tests/test_no_leakage.py` and adapt:

```python
import pandas as pd
import pytest

@pytest.fixture
def splits():
    return load_splits()  # returns train, val, test DataFrames

TIME_COL = "event_datetime"   # change to your temporal column
TARGET   = "target"           # change to your target column
DENYLIST: set[str] = set()    # fill in: any column that is a function of the target

def test_temporal_ordering(splits):
    train, val, test = splits
    assert train[TIME_COL].max() < val[TIME_COL].min()
    assert val[TIME_COL].max()   < test[TIME_COL].min()

def test_no_high_corr_features(splits):
    train, _, _ = splits
    corr = train.drop(columns=[TARGET]).select_dtypes("number").corrwith(train[TARGET]).abs()
    assert (corr > 0.95).sum() == 0, corr[corr > 0.95].to_dict()

def test_no_target_proxy_in_features():
    # Fill DENYLIST with your domain's target-component columns.
    feats = set(load_feature_list())
    assert feats.isdisjoint(DENYLIST), f"Leak features present: {feats & DENYLIST}"

def test_rolling_window_is_left_closed(monkeypatch):
    # If you use df.rolling(...) anywhere, grep for closed='left'.
    import subprocess
    out = subprocess.check_output(["grep", "-rn", "rolling(", "src/"]).decode()
    for line in out.splitlines():
        if "closed=" not in line:
            pytest.fail(f"Rolling without explicit closed=: {line}")
        assert "closed='left'" in line or 'closed="left"' in line, line

def test_scaler_fit_only_on_train():
    # Static check: any .fit() call on a preprocessor outside the train pipeline is suspect.
    # Use code review for this one; no clean automated test.
    pass
```

## When you find a leak

1. Don't silently drop the feature — log it. Add a row to `docs/leakage_findings.md` with the date, feature, mechanism, and how it was caught.
2. Re-run the affected experiment and tag the bad runs in MLflow with `leakage_found=true` so they're filtered from comparisons.
3. If a leak made it to a deployed model, that's an incident — write a short postmortem.

## Anti-patterns to refuse

- `train_test_split(df, random_state=42)` on time-series data. Always temporal.
- Computing feature statistics on `df` and then splitting.
- `df['mean_fare_by_zone'] = df.groupby('zone')['fare'].transform('mean')` (in-place groupby on full data).
- Using `LabelEncoder` on the categorical without persisting the mapping, so test rows with unseen categories silently get encoded as the same value as train's most-frequent class.
