# Preprocessing Overview

## Model-Agnostic vs Model-Specific

The preprocessing architecture separates concerns into two clear phases:

### Model-Agnostic Preprocessing (DataProcessor)

Operations that apply regardless of which ML model you use:

- ✅ Data cleaning and validation
- ✅ Feature engineering
- ✅ Train/validation/test splitting (70/15/15)
- ✅ Correlation-based feature selection

**Why Model-Agnostic?**
- Same preprocessing for Ridge, RandomForest, XGBoost, etc.
- No assumptions about model requirements
- Prevents tight coupling between data and models

### Model-Specific Preprocessing (ModelTrainer)

Operations that depend on the chosen model:

- ✅ Missing value imputation (median vs mode)
- ✅ Feature scaling (StandardScaler)
- ✅ Power transformations (handle skewness)
- ✅ Target transformation (log for skewed targets)

**Why Model-Specific?**
- Different models have different requirements
- Linear models need scaling, tree models don't
- Transformations must be fitted only on training data

## Preprocessing Flow Diagram

```mermaid
flowchart TD
    Start([Raw CSV]) --> Phase1

    subgraph Phase1["Phase 1: Load and Clean"]
        Load[Load CSV]
        PKClean[Clean Primary Key]
        ForceNumeric[Force Numeric]
        BusinessRules[Apply Business Rules]
        NormalizeLDA[Normalize LDA]

        Load --> PKClean
        PKClean --> ForceNumeric
        ForceNumeric --> BusinessRules
        BusinessRules --> NormalizeLDA
    end

    Phase1 --> Phase2

    subgraph Phase2["Phase 2: Feature Engineering"]
        SepXy[Separate X and y]
        DropCols[Drop Non-Predictive]
        Classify[Classify Binary/Non-Binary]

        SepXy --> DropCols
        DropCols --> Classify
    end

    Phase2 --> Phase3

    subgraph Phase3["Phase 3: Split Data"]
        Split1[Train 70% vs Temp 30%]
        Split2[Val 15% vs Test 15%]

        Split1 --> Split2
    end

    Phase3 --> Phase4

    subgraph Phase4["Phase 4: Handle Correlation"]
        CalcCorr[Calculate on Train Only]
        DropHighCorr[Drop High Corr Features]
        ApplyAll[Apply to Val/Test]

        CalcCorr --> DropHighCorr
        DropHighCorr --> ApplyAll
    end

    Phase4 --> Output([Clean Splits])

    style Phase1 fill:#a8dadc
    style Phase2 fill:#64b5f6
    style Phase3 fill:#90caf9
    style Phase4 fill:#e63946
```

## Data Leakage Prevention

!!! danger "Critical"
    All statistics (correlation, mean, std) are calculated **only** on the training set, then applied to validation and test sets.

**Example**:
```python
# CORRECT: Fit on train only
corr_matrix = X_train.corr()
high_corr_cols = find_high_corr(corr_matrix, threshold=0.9)

# Apply to all splits
X_train = X_train.drop(columns=high_corr_cols)
X_val = X_val.drop(columns=high_corr_cols)
X_test = X_test.drop(columns=high_corr_cols)

# WRONG: Fit on all data
corr_matrix = pd.concat([X_train, X_val, X_test]).corr()  # DATA LEAKAGE!
```

## Key Classes

- **[DataProcessor](data-processor.md)**: Main orchestrator for preprocessing pipeline
- **[DataCleaner](data-cleaner.md)**: Method chaining for data cleaning
- **[Utilities](utilities.md)**: Helper classes (DataExplorer, DataLoader, etc.)

## Next Steps

- [DataProcessor Details](data-processor.md)
- [Data Flow](../architecture/data-flow.md)
- [Model Training](../modeling/model-trainer.md)
