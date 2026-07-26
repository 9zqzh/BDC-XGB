---
name: Model Hyperparameter Tuning
description: Optimize hyperparameters using grid search, random search, Bayesian optimization, and automated ML frameworks like Optuna and Hyperopt
---

# Model Hyperparameter Tuning

## Overview

Hyperparameter tuning is the process of systematically searching for the best combination of model configuration parameters to maximize performance on validation data.

## When to Use

- When optimizing model performance beyond baseline configurations
- When comparing different parameter combinations systematically
- When fine-tuning complex models with many hyperparameters
- When seeking the best trade-off between bias, variance, and training time
- When improving model generalization on validation and test data
- When exploring parameter spaces for neural networks, tree models, or ensemble methods

## Tuning Methods

- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter space
- **Bayesian Optimization**: Probabilistic model-based search
- **Hyperband**: Multi-fidelity optimization
- **Evolutionary Algorithms**: Genetic algorithm based search
- **Population-based Training**: Distributed parameter optimization

## Hyperparameters by Model Type

- **Tree Models**: max_depth, min_samples_split, learning_rate
- **Neural Networks**: learning_rate, batch_size, num_layers, dropout
- **SVM**: C, kernel, gamma
- **Ensemble**: n_estimators, max_features, min_samples_leaf

## Python Implementation

### 1. Grid Search with scikit-learn
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### 2. Random Search
```python
param_dist = {
    'n_estimators': np.arange(50, 300, 10),
    'max_depth': np.arange(5, 30, 1),
    'min_samples_split': np.arange(2, 20, 1),
    'min_samples_leaf': np.arange(1, 10, 1)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42
)
```

### 3. Bayesian Optimization with Optuna
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
print(f"Best trial: {study.best_trial.params}")
print(f"Best score: {study.best_trial.value:.4f}")
```

### 4. Learning Rate Tuning for Neural Networks (PyTorch)
```python
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
results = {}

for lr in learning_rates:
    model = SimpleNN(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train_scaled))
        loss = criterion(outputs, torch.FloatTensor(y_train).unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_scaled))
        acc = ((preds > 0).float() == torch.FloatTensor(y_test).unsqueeze(1)).float().mean()
    results[lr] = acc.item()
```

### 5. Hyperparameter Importance Analysis
```python
import optuna.visualization as vis

# Parameter importance
importance = vis.plot_param_importances(study)
importance.show()

# Optimization history
history = vis.plot_optimization_history(study)
history.show()

# Slice plot
slice_plot = vis.plot_slice(study, params=['n_estimators', 'max_depth'])
slice_plot.show()
```

### 6. Cross-Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(
    RandomForestClassifier(**best_params, random_state=42),
    X_train, y_train, cv=cv_strategy, scoring=scoring
)

for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 7. Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(**best_params, random_state=42),
    X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 8. Validation Curve
```python
from sklearn.model_selection import validation_curve

param_range = [5, 10, 15, 20, 25, 30]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train, param_name='max_depth', param_range=param_range,
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Training')
plt.plot(param_range, val_scores.mean(axis=1), 'o-', label='Validation')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve for max_depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

- Start with a small random search to understand the landscape
- Use Bayesian optimization for medium parameter spaces
- Scale to distributed search for massive spaces
- Monitor for overfitting with validation curves
- Consider training time vs performance trade-offs
- Use early stopping when possible
- Log all trials for reproducibility
- Use cross-validation consistently
- Separate tuning data from final test data
- Document selected hyperparameters and rationale

## Key Metrics to Monitor

- **Validation score**: Primary optimization target
- **Training time**: Computational cost
- **Convergence speed**: Iterations to optimum
- **Parameter sensitivity**: Stability of results
- **Generalization gap**: Train vs validation difference
- **Model complexity proxy**: Number or norm of parameters
