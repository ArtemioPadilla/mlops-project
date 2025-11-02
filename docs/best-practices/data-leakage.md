# Preventing Data Leakage

Critical practices to prevent data leakage in ML pipelines.

## What is Data Leakage?

Data leakage occurs when information from outside the training dataset influences the model.

## Common Sources

### 1. Correlation Calculation

❌ **Wrong**: Calculate on all data
```python
# BAD: Uses test data
all_data = pd.concat([X_train, X_val, X_test])
corr = all_data.corr()
```

✅ **Correct**: Calculate on train only
```python
# GOOD: Train only
corr = X_train.corr()
# Apply to all
X_train = X_train.drop(columns=high_corr_cols)
X_val = X_val.drop(columns=high_corr_cols)
X_test = X_test.drop(columns=high_corr_cols)
```

### 2. Scaling/Normalization

❌ **Wrong**: Fit on all data
```python
scaler = StandardScaler()
scaler.fit(pd.concat([X_train, X_test]))  # LEAKAGE!
```

✅ **Correct**: Fit on train, apply to test
```python
scaler = StandardScaler()
scaler.fit(X_train)  # Learn from train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## How This Project Prevents Leakage

### DataProcessor

- ✅ Correlation: Train-only calculation
- ✅ Splitting: Before any transformations
- ✅ Feature selection: Train-only statistics

### ModelTrainer

- ✅ sklearn Pipeline: Fits only on train data
- ✅ ColumnTransformer: Learns from train, applies to val/test
- ✅ Imputation: Train statistics only

## Verification

Check for leakage:
```python
# Val and test should have slightly different distributions
print(X_train.mean())
print(X_test.mean())  # Should differ slightly
```
