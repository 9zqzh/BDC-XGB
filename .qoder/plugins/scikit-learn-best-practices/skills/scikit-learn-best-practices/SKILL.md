---
name: scikit-learn-best-practices
description: Best practices for scikit-learn machine learning, model development, evaluation, and deployment in Python
---

# Scikit-learn Best Practices

Expert guidelines for scikit-learn development, focusing on machine learning workflows, model development, evaluation, and best practices.

## Code Style and Structure

- Write concise, technical responses with accurate Python examples
- Prioritize reproducibility in machine learning workflows
- Use functional programming for data pipelines
- Use object-oriented programming for custom estimators
- Prefer vectorized operations over explicit loops
- Follow PEP 8 style guidelines

## Machine Learning Workflow

### Data Preparation

- Always split data before any preprocessing: train/validation/test
- Use `train_test_split()` with `random_state` for reproducibility
- Stratify splits for imbalanced classification: `stratify=y`
- Keep test set completely separate until final evaluation

### Feature Engineering

- Scale features appropriately for distance-based algorithms
- Use `StandardScaler` for normally distributed features
- Use `MinMaxScaler` for bounded features
- Use `RobustScaler` for data with outliers
- Encode categorical variables: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`
- Handle missing values: `SimpleImputer`, `KNNImputer`

### Pipelines

- Always use `Pipeline` to chain preprocessing and modeling
- Prevents data leakage by fitting transformers only on training data
- Makes code cleaner and more reproducible
- Enables easy deployment and serialization

### Column Transformers

- Use `ColumnTransformer` for different preprocessing per feature type
- Combine numeric and categorical preprocessing in single pipeline

## Model Selection and Tuning

### Cross-Validation

- Use cross-validation for reliable performance estimates
- Use appropriate CV strategy: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `GroupKFold`

### Hyperparameter Tuning

- Use `GridSearchCV` for exhaustive search
- Use `RandomizedSearchCV` for large parameter spaces
- Always tune on training/validation data, never test data

## Model Evaluation

- Use appropriate metrics for your problem
- Report confidence intervals, not just point estimates
- Use multiple metrics to understand model behavior
- Compare against meaningful baselines

## Key Conventions

- Import from submodules: `from sklearn.ensemble import RandomForestClassifier`
- Set `random_state` for reproducibility
- Use pipelines to prevent data leakage
- Document model choices and hyperparameters
