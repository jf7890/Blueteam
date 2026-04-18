# Gatekeeper ML Usage Guide

## Overview

`gatekeeper_ml` is a lightweight HTTP payload screening module designed for security pipelines that need very fast binary classification of raw string inputs.

It classifies each input as:

- `0`: Normal
- `1`: Suspicious

The module is intended to sit in front of a more expensive analysis stage such as an LLM, RAG pipeline, sandbox, or deep inspection service. Its purpose is to quickly filter out clearly benign values and forward only higher-risk inputs for deeper review.

## Primary Use Cases

Typical inputs include:

- request paths and URIs
- query parameter values
- header values
- cookie values
- form fields
- decoded fragments from HTTP access or security logs

This module is especially useful in:

- Blue Team agent workflows
- SIEM enrichment pipelines
- WAF analytics pipelines
- pre-filtering stages for incident-response automation

## Installation

From the project root:

```bash
pip install -e .
```

This installs the package in editable mode for local development and integration testing.

## Quick Start

```python
from gatekeeper_ml import PayloadClassifier

classifier = PayloadClassifier()
predictions = classifier.predict_batch([
    "/api/v1/users/42",
    "<script>alert(1)</script>",
    "' OR 1=1 --",
])

print(predictions)
# Example: [0, 1, 1]
```

## Model Loading Behavior

`PayloadClassifier` resolves the model in the following order:

1. `model_path` passed directly to the constructor
2. `GATEKEEPER_MODEL_PATH` environment variable
3. packaged default model at `src/gatekeeper_ml/models/payload_classifier.pkl`

Examples:

```python
from gatekeeper_ml import PayloadClassifier

# Use the packaged default model
default_classifier = PayloadClassifier()

# Use a custom model path
custom_classifier = PayloadClassifier(model_path="models/gatekeeper_payload_classifier.pkl")
```

Environment-based loading:

```bash
export GATEKEEPER_MODEL_PATH="/absolute/path/to/payload_classifier.pkl"
```

## Python API

### `PayloadClassifier`

Main inference entrypoint for the package.

#### Constructor

```python
PayloadClassifier(model_path: str | None = None)
```

Arguments:

- `model_path`: Optional explicit path to a trained `.pkl` artifact

Behavior:

- loads the model once
- reuses the same instance for the same resolved model path
- applies conservative regex-based fast paths before invoking the ML model

#### `predict_batch`

```python
predict_batch(inputs: list[str]) -> list[int]
```

Arguments:

- `inputs`: a batch of raw strings to classify

Returns:

- a list of binary predictions matching the input order

Example:

```python
results = classifier.predict_batch([
    "12345",
    "550e8400-e29b-41d4-a716-446655440000",
    "../../etc/passwd",
])
```

## Fast-Path Heuristics

To reduce inference cost, the classifier first short-circuits obviously benign inputs such as:

- empty strings
- numeric IDs
- UUIDs
- standard safe header names
- simple safe URI patterns

These heuristics are intentionally conservative. If an input contains suspicious markers, the classifier falls back to the trained model instead of trusting the shortcut.

## Command-Line Usage

The repository also includes a CLI for dataset preparation, training, and batch prediction.

### Fetch Training Data

```bash
python cli.py fetch
```

### Train the Model

```bash
python cli.py train
```

### Predict From the CLI

```bash
python cli.py predict "/api/v1/users/42" "<script>alert(1)</script>"
```

## Training Outputs

After training, the following artifacts are typically produced under `models/`:

- trained model artifact
- training metrics JSON
- evaluation report
- confusion matrix plot
- ROC curve plot
- feature importance plot

These outputs support operational validation and model review before deployment.

## Integration Example for a Blue Team Agent

```python
from gatekeeper_ml import PayloadClassifier


def filter_log_components(values: list[str]) -> list[str]:
    classifier = PayloadClassifier()
    predictions = classifier.predict_batch(values)
    return [
        value
        for value, label in zip(values, predictions)
        if label == 1
    ]


components = [
    "/healthz",
    "sessionid=4f8b5d9c0a1e2f3d4c5b6a7980112233",
    "<svg/onload=alert(1)>",
]

suspicious_only = filter_log_components(components)
print(suspicious_only)
```

In a larger pipeline, the returned suspicious values can then be:

- sent to an LLM for explanation
- attached to an alert ticket
- stored for triage
- correlated with other telemetry

## Error Handling

If no model file is found, `PayloadClassifier` raises a clear `FileNotFoundError` describing the resolved path and the supported loading options.

Recommended handling pattern:

```python
from gatekeeper_ml import PayloadClassifier

try:
    classifier = PayloadClassifier()
except FileNotFoundError as exc:
    print(f"Model load failed: {exc}")
```

## Operational Recommendations

- Use batch prediction whenever possible to reduce per-item overhead.
- Retrain periodically as payload patterns evolve.
- Validate the model against real benign traffic from your environment before production rollout.
- Keep a custom model artifact if your logs contain organization-specific patterns.
- Treat this module as a fast filter, not a complete detection system.

## Summary

`gatekeeper_ml` is best used as a low-latency security gate that screens raw HTTP-related strings before a more expensive downstream analysis stage. It is simple to embed, easy to retrain, and designed for practical agent and SOC automation workflows.
