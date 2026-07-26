---
name: mle-workflow
description: Production machine-learning engineering workflow for data contracts, reproducible training, model evaluation, deployment, monitoring, and rollback. Use when building, reviewing, or hardening ML systems beyond one-off notebooks.
license: MIT
metadata:
  origin: ECC
---

# Machine Learning Engineering Workflow

Use this skill to turn model work into a production ML system with clear data contracts, repeatable training, measurable quality gates, deployable artifacts, and operational monitoring.

## When to Activate

- Planning or reviewing a production ML feature, model refresh, ranking system, recommender, classifier, embedding workflow, or forecasting pipeline
- Converting notebook code into a reusable training, evaluation, batch inference, or online inference pipeline
- Designing model promotion criteria, offline/online evals, experiment tracking, or rollback paths
- Debugging failures caused by data drift, label leakage, stale features, artifact mismatch, or inconsistent training and serving logic
- Adding model monitoring, canary rollout, shadow traffic, or post-deploy quality checks

## Scope Calibration

Use only the lanes that fit the system in front of you. This skill is useful for ranking, search, recommendations, classifiers, forecasting, embeddings, LLM workflows, anomaly detection, and batch analytics, but it should not force one architecture onto all of them.

- Do not assume every model has supervised labels, online serving, a feature store, PyTorch, GPUs, human review, A/B tests, or real-time feedback.
- Do not add heavyweight MLOps machinery when a data contract, baseline, eval script, and rollback note would make the change reviewable.
- Do make assumptions explicit when the project lacks labels, delayed outcomes, slice definitions, production traffic, or monitoring ownership.
- Treat examples as interchangeable scaffolds. Replace metrics, serving mode, data stores, and rollout mechanics with the project-native equivalents.

## Related Skills

- `python-patterns` and `python-testing` for Python implementation and pytest coverage
- `pytorch-patterns` for deep learning models, data loaders, device handling, and training loops
- `eval-harness` and `ai-regression-testing` for promotion gates and agent-assisted regression checks
- `database-migrations`, `postgres-patterns`, and `clickhouse-io` for data storage and analytics surfaces
- `deployment-patterns`, `docker-patterns`, and `security-review` for serving, secrets, containers, and production hardening

## Reuse the SWE Surface

Do not treat MLE as separate from software engineering. Most ECC SWE workflows apply directly to ML systems, often with stricter failure modes:

| SWE surface | MLE use |
|-------------|---------|
| `product-capability` / `architecture-decision-records` | Turn model work into explicit product contracts and record irreversible data, model, and rollout choices |
| `codebase-onboarding` / `code-tour` | Find existing training, feature, serving, eval, and monitoring paths before introducing a parallel ML stack |
| `plan` / `feature-dev` | Scope model changes as product capabilities with data, eval, serving, and rollback phases |
| `tdd-workflow` / `python-testing` | Test feature transforms, split logic, metric calculations, artifact loading, and inference schemas |
| `code-reviewer` / `mle-reviewer` | Review code quality plus ML-specific leakage, reproducibility, promotion, and monitoring risks |
| `eval-harness` / `verification-loop` | Turn offline metrics, slice checks, latency budgets, and rollback drills into repeatable gates |
| `ai-regression-testing` | Preserve every production bug as a regression: missing feature, stale label, bad artifact, schema drift, or serving mismatch |
| `api-design` / `backend-patterns` | Design prediction APIs, batch jobs, idempotent retraining endpoints, and response envelopes |
| `deployment-patterns` / `docker-patterns` | Package reproducible training and serving images with health checks, resource limits, and rollback |
| `canary-watch` / `dashboard-builder` | Make rollout health visible with model-version, slice, drift, latency, cost, and delayed-label dashboards |
| `security-review` / `security-scan` | Check model artifacts, notebooks, prompts, datasets, and logs for secrets, PII, unsafe deserialization, and supply-chain risk |
| `e2e-testing` / `browser-qa` / `accessibility` | Test critical product flows that consume predictions, including explainability and fallback UI states |
| `benchmark` / `performance-optimizer` | Measure throughput, p95 latency, memory, GPU utilization, and cost per prediction or retrain |
| `cost-aware-llm-pipeline` | Route LLM/embedding workloads by quality, latency, and budget instead of defaulting to the largest model |

## Iteration Compact

Before touching model code, compress the work into one reviewable artifact:

```text
Goal:
Who cares:
Decision owner:
User or system action changed by the model:
Success metric:
Guardrail metrics:
Mistake budget:
Unacceptable mistakes:
Acceptable mistakes:
Assumptions:
Constraints:
Labels and data snapshot:
Baseline:
Candidate signals:
Threshold or config plan:
Eval slices:
Known risks:
Next experiment:
Rollback or fallback:
```

## Decision Brain

1. Start from the decision, not the model. Name the action that changes downstream behavior.
2. Name who cares and why. Different stakeholders pay different costs for false positives, false negatives, latency, compute spend, opacity, or missed opportunities.
3. Convert ambiguity into hypotheses. Ask what signal would separate outcomes, what evidence would disprove it, and what simple baseline should be hard to beat.
4. Research prior art or a nearby known problem before inventing a bespoke system.
5. Score choices with `(probability, confidence) x (cost, severity, importance, impact)`.
6. Consider adversarial behavior, incentives, selective disclosure, distribution shift, and feedback loops.
7. Prefer the simplest change that reduces the most important mistake.
8. Capture the decision, evidence, counterargument, and next reversible step.

## Metric and Mistake Economics

- Use a confusion matrix early so the team can discuss concrete false positives and false negatives instead of abstract accuracy.
- Favor precision when the cost of an incorrect positive decision dominates.
- Favor recall when the cost of a missed positive dominates.
- Use F1 only when the precision/recall tradeoff is genuinely balanced and explainable.
- Use AUC or ranking metrics when ordering quality matters more than a single threshold.
- Track latency, throughput, memory, and cost as first-class metrics.
- Compare against a baseline and the current production model before celebrating an offline gain.

## Core Workflow

### 1. Define the Prediction Contract

Capture the product-level contract before writing model code:
- Prediction target and decision owner
- Input entity, output schema, confidence/calibration fields, and allowed latency
- Batch, online, streaming, or hybrid serving mode
- Fallback behavior when the model, feature store, or dependency is unavailable
- Human review or override path for high-impact decisions

### 2. Lock the Data Contract

Every ML task needs an explicit data contract:
- Entity grain and primary key
- Label definition, label timestamp, and label availability delay
- Feature timestamp, freshness SLA, and point-in-time join rules
- Train, validation, test, and backtest split policy
- Required columns, allowed nulls, ranges, categories, and units
- Dataset version or snapshot ID for reproducibility

### 3. Build a Reproducible Pipeline

- Use typed config files or dataclasses for all hyperparameters and paths
- Pin package and model dependencies
- Set random seeds and document any nondeterministic GPU behavior
- Record dataset version, code SHA, config hash, metrics, and artifact URI
- Save preprocessing logic with the model artifact

### 4. Evaluate Before Promotion

Promotion criteria should be declared before training finishes:
- Baseline model and current production model comparison
- Primary metric aligned to product behavior
- Guardrail metrics for latency, calibration, fairness slices, cost, and error concentration
- Slice metrics for important cohorts, geographies, devices, languages, or data sources
- Explicit "do not ship" thresholds

### 5. Package for Serving

- Model artifact includes version, training data reference, config, and preprocessing
- Input schema rejects invalid, stale, or out-of-range features
- Output schema includes model version and confidence or explanation fields when useful
- Serving path has timeout, batching, resource limits, and fallback behavior

### 6. Operate the Model

- Availability, error rate, timeout rate, queue depth, and p50/p95/p99 latency
- Feature null rate, range drift, categorical drift, and freshness drift
- Prediction distribution drift and confidence distribution drift
- Label arrival health and delayed quality metrics
- Business KPI guardrails and rollback triggers

## Review Checklist

- [ ] Prediction contract is explicit and testable
- [ ] Data contract defines entity grain, label timing, feature timing, and snapshot/version
- [ ] Leakage risks were checked against prediction-time availability
- [ ] Training is reproducible from code, config, data version, and seed
- [ ] Metrics compare against baseline and current production model
- [ ] Slice metrics and guardrails are included for high-risk cohorts
- [ ] Promotion gates are automated and fail closed
- [ ] Training and serving transformations are shared or equivalence-tested
- [ ] Serving path validates inputs and has timeout, fallback, and rollback behavior
- [ ] Monitoring covers system health, feature drift, prediction drift, and delayed labels

## Anti-Patterns

- Notebook state is required to reproduce the model
- Random split leaks future data into validation or test sets
- Feature joins ignore event time and label availability
- Offline metric improves while important slices regress
- Thresholds are tuned on the test set repeatedly
- Training preprocessing is copied manually into serving code
- Model version is missing from prediction logs
- Monitoring only checks service uptime, not data or prediction quality
- Rollback requires retraining instead of switching to a known-good artifact
---
name: mle-workflow
description: Production machine-learning engineering workflow for data contracts, reproducible training, model evaluation, deployment, monitoring, and rollback. Use when building, reviewing, or hardening ML systems beyond one-off notebooks.
license: MIT
metadata:
  origin: ECC
---

# Machine Learning Engineering Workflow

Use this skill to turn model work into a production ML system with clear data contracts, repeatable training, measurable quality gates, deployable artifacts, and operational monitoring.

## When to Activate

- Planning or reviewing a production ML feature, model refresh, ranking system, recommender, classifier, embedding workflow, or forecasting pipeline
- Converting notebook code into a reusable training, evaluation, batch inference, or online inference pipeline
- Designing model promotion criteria, offline/online evals, experiment tracking, or rollback paths
- Debugging failures caused by data drift, label leakage, stale features, artifact mismatch, or inconsistent training and serving logic
- Adding model monitoring, canary rollout, shadow traffic, or post-deploy quality checks

## Scope Calibration

Use only the lanes that fit the system in front of you. This skill is useful for ranking, search, recommendations, classifiers, forecasting, embeddings, LLM workflows, anomaly detection, and batch analytics, but it should not force one architecture onto all of them.

- Do not assume every model has supervised labels, online serving, a feature store, PyTorch, GPUs, human review, A/B tests, or real-time feedback.
- Do not add heavyweight MLOps machinery when a data contract, baseline, eval script, and rollback note would make the change reviewable.
- Do make assumptions explicit when the project lacks labels, delayed outcomes, slice definitions, production traffic, or monitoring ownership.
- Treat examples as interchangeable scaffolds. Replace metrics, serving mode, data stores, and rollout mechanics with the project-native equivalents.
