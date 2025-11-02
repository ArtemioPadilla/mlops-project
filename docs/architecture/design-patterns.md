# Design Patterns

This page explains the key design patterns used in the MLOps system.

## Pattern 1: Builder Pattern (Method Chaining)

**Used in**: `DataCleaner`

**Purpose**: Create a fluent interface for building a cleaning pipeline.

### Implementation

```python
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_primary_key(self, key: str) -> 'DataCleaner':
        # Cleaning logic
        return self  # Return self for chaining

    def force_numeric(self, exclude: List[str] = None) -> 'DataCleaner':
        # Convert to numeric
        return self  # Return self

    def apply_business_rules(self) -> 'DataCleaner':
        # Apply rules
        return self

    def get_df(self) -> pd.DataFrame:
        return self.df
```

### Usage

```python
cleaner = DataCleaner(df)
cleaned_df = (cleaner
    .clean_primary_key(key="url")
    .force_numeric(exclude=["url"])
    .apply_business_rules()
    .normalize_lda(["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"])
    .get_df())
```

**Benefits**:
- ✅ Readable, declarative code
- ✅ Easy to add/remove steps
- ✅ Maintains immutability (copies DataFrame)

---

## Pattern 2: Strategy Pattern

**Used in**: `ModelTrainer`

**Purpose**: Allow different ML algorithms to be swapped without changing the training code.

### Implementation

```python
class ModelTrainer:
    def __init__(self, data_processor, estimator, model_name):
        self.estimator = estimator  # Strategy: any sklearn estimator
        # ...

    def train_model(self):
        # Works with ANY sklearn estimator
        self.pipeline.fit(self.X_train, self.y_train_transformed)
```

### Usage

```python
# Strategy 1: Ridge Regression
trainer1 = ModelTrainer(processor, Ridge(alpha=1.0), "Ridge")

# Strategy 2: Random Forest
trainer2 = ModelTrainer(processor, RandomForestRegressor(n_estimators=100), "RF")

# Strategy 3: XGBoost
trainer3 = ModelTrainer(processor, XGBRegressor(), "XGBoost")

# All use the same training interface
for trainer in [trainer1, trainer2, trainer3]:
    trainer.train_model()
    metrics = trainer.evaluate_model()
```

**Benefits**:
- ✅ Easy to experiment with different models
- ✅ Open/Closed Principle (open for extension, closed for modification)
- ✅ No code changes needed to add new models

---

## Pattern 3: Template Method Pattern

**Used in**: `DataProcessor.process()`

**Purpose**: Define the skeleton of an algorithm, allowing subclasses to override specific steps.

### Implementation

```python
class DataProcessor:
    def process(self):
        """Template method defining the algorithm structure."""
        # Step 1: Load and clean (can be customized)
        df = self.load_and_clean()

        # Step 2: Feature engineering (can be customized)
        X, y = self.engineer_features(df)

        # Step 3: Split data (fixed algorithm)
        self.split_data(X, y)

        # Step 4: Handle correlation (can be customized)
        self._handle_high_correlation()

        return self
```

### Customization Points

```python
class DataProcessor:
    def load_and_clean(self):
        """Hook method - can be overridden."""
        cleaner = DataCleaner(pd.read_csv(self.filepath))
        # Default cleaning logic
        return cleaner.get_df()

    def engineer_features(self, df):
        """Hook method - can be overridden."""
        # Default feature engineering
        return X, y
```

**Benefits**:
- ✅ Enforces consistent workflow
- ✅ Allows customization at specific points
- ✅ Reduces code duplication

---

## Pattern 4: Facade Pattern

**Used in**: CLI modules

**Purpose**: Provide a simplified interface to a complex subsystem.

### Implementation

```python
# Complex subsystem
from mlops_online_news_popularity.preprocessing import (
    DataProcessor, DataCleaner, DataExplorer, DataLoader
)

# Facade: Simple interface
@app.command()
def main(input_path, output_dir, ...):
    # Hide complexity behind simple method calls
    processor = DataProcessor(filepath=input_path, ...)
    processor.process()  # Single call hides multiple steps

    # Save outputs
    for name, df in splits.items():
        df.to_csv(output_dir / f"{name}.csv")
```

### User Experience

```bash
# User doesn't need to know about DataCleaner, train_test_split, etc.
make preprocess

# vs manually
from mlops_online_news_popularity.preprocessing import DataCleaner, DataLoader
loader = DataLoader()
df = loader.load_csv("...")
cleaner = DataCleaner(df)
# ... many more steps
```

**Benefits**:
- ✅ Simplifies complex operations
- ✅ Reduces learning curve
- ✅ Provides reasonable defaults

---

## Pattern 5: Dependency Injection

**Used in**: `ModelTrainer`

**Purpose**: Inject dependencies rather than creating them internally.

### Implementation

```python
class ModelTrainer:
    def __init__(self, data_processor: DataProcessor, estimator, model_name: str):
        # Dependencies injected, not created
        self.data_processor = data_processor  # Injected
        self.estimator = estimator              # Injected

        # Use injected dependencies
        self.X_train = data_processor.X_train
        self.y_train = data_processor.y_train
```

### Benefits

**Without DI** (tight coupling):
```python
class ModelTrainer:
    def __init__(self, filepath):
        # Creates its own dependencies - hard to test
        self.processor = DataProcessor(filepath)
        self.processor.process()
```

**With DI** (loose coupling):
```python
# Easy to test with mock
mock_processor = Mock(DataProcessor)
mock_processor.X_train = test_data
trainer = ModelTrainer(mock_processor, Ridge(), "Test")
```

**Benefits**:
- ✅ Easier testing (can inject mocks)
- ✅ Loose coupling
- ✅ Reusable components

---

## Pattern 6: Composition Over Inheritance

**Used Throughout**

**Principle**: Favor object composition over class inheritance.

### Implementation

```python
class DataProcessor:
    def __init__(self, ...):
        # Composition: "has-a" relationship
        self.cleaner = None  # Will contain DataCleaner
        self.explorer = None  # Will contain DataExplorer

    def load_and_clean(self):
        # Uses composition
        cleaner = DataCleaner(df)  # Creates and uses, doesn't inherit
        return cleaner.clean_primary_key(...).get_df()
```

### vs Inheritance

```python
# Bad: Inheritance creates tight coupling
class DataProcessor(DataCleaner, DataExplorer):  # DON'T DO THIS
    pass

# Good: Composition allows flexibility
class DataProcessor:
    def __init__(self):
        self.cleaner = DataCleaner(...)  # Can swap implementations
        self.explorer = DataExplorer()
```

**Benefits**:
- ✅ More flexible than inheritance
- ✅ Avoids diamond problem
- ✅ Easier to test and mock

---

## Pattern 7: Pipeline Pattern (sklearn)

**Used in**: `ModelTrainer`

**Purpose**: Chain transformations to ensure proper train/test handling.

### Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Build pipeline
preprocessor = ColumnTransformer([...])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', estimator)
])

# Fit on train data only
pipeline.fit(X_train, y_train)

# Transform and predict (uses learned parameters)
y_pred = pipeline.predict(X_test)
```

**Benefits**:
- ✅ Prevents data leakage
- ✅ Reproducible transformations
- ✅ Easy deployment (one object)

---

## Pattern 8: Factory Method (Implicit)

**Used in**: `Experimento`

**Purpose**: Create objects based on configuration.

### Implementation

```python
class Experimento:
    def _create_estimator(self, class_path: str, params: dict):
        """Factory method to create estimators from config."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        estimator_class = getattr(module, class_name)
        return estimator_class(**params)  # Create instance

# Usage from YAML config
models_to_try:
  Ridge:
    class_path: "sklearn.linear_model.Ridge"
    alpha: 1.0

# Experimento creates the right object
estimator = self._create_estimator("sklearn.linear_model.Ridge", {"alpha": 1.0})
```

**Benefits**:
- ✅ Configuration-driven
- ✅ No code changes for new models
- ✅ Centralized object creation

---

## Anti-Patterns Avoided

### ❌ God Object

**Avoided**: Not putting everything in one class.

**How**: Separate concerns into `DataProcessor`, `DataCleaner`, `ModelTrainer`, etc.

### ❌ Circular Dependencies

**Avoided**: Clear dependency hierarchy.

**How**: config → utils → preprocessing → modeling → cli

### ❌ Magic Numbers/Strings

**Avoided**: Hardcoded values scattered in code.

**How**: Centralize in `config.py` and YAML files.

---

## Design Principles

### SOLID Principles Applied

**S - Single Responsibility**:
- `DataCleaner`: Only cleaning
- `DataProcessor`: Only preprocessing orchestration
- `ModelTrainer`: Only model training

**O - Open/Closed**:
- Open for extension (add new models via config)
- Closed for modification (don't change core code)

**L - Liskov Substitution**:
- Any sklearn estimator can be used in `ModelTrainer`

**I - Interface Segregation**:
- Small, focused interfaces (`.process()`, `.train_model()`)

**D - Dependency Inversion**:
- Depend on abstractions (sklearn estimator interface)
- Not on concrete implementations

---

## Next Steps

- [Component Interactions](components.md)
- [Data Flow](data-flow.md)
- [Model Trainer Implementation](../modeling/model-trainer.md)
