---
title: "Getting Started with MLOps: A Practical Guide"
date: 2025-12-01
draft: true
tags: ["MLOps", "DevOps", "Machine Learning", "Infrastructure"]
categories: ["MLOps", "Tutorial"]
description: "Learn the fundamentals of MLOps and how to build production-ready ML systems"
cover:
  image: "https://images.unsplash.com/photo-1667372393119-3d4c48d07fc9?w=1200&q=80"
  alt: "MLOps pipeline infrastructure"
  caption: "Building production-ready ML systems"
---

## What is MLOps?

MLOps (Machine Learning Operations) bridges the gap between ML development and production deployment. It's about taking models from notebooks to production reliably, efficiently, and at scale.

## Why MLOps Matters

Without MLOps:
- Models take months to deploy
- Performance degrades silently
- Experiments aren't reproducible
- Teams work in silos

With MLOps:
- Deploy models in days, not months
- Automated monitoring catches issues early
- Full experiment tracking and reproducibility
- Seamless collaboration

## The MLOps Stack

### 1. Version Control

**Code**: Git (GitHub, GitLab, Bitbucket)

**Data**: DVC, LakeFS, or Delta Lake

```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add raw dataset"

# Push data to remote storage
dvc push
```

### 2. Experiment Tracking

**Tools**: MLflow, Weights & Biases, Neptune

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

mlflow.set_experiment("customer_churn")

with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    params = {'n_estimators': 100, 'max_depth': 10}
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Log metrics
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### 3. Model Registry

Central repository for production models:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri, "churn_predictor")

# Transition to production
client.transition_model_version_stage(
    name="churn_predictor",
    version=1,
    stage="Production"
)
```

### 4. Model Serving

**Options**:
- REST API (FastAPI, Flask)
- Batch predictions (Airflow, Prefect)
- Real-time streaming (Kafka + Flink)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()

# Load model from registry
model = mlflow.pyfunc.load_model("models:/churn_predictor/Production")

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction"""
    prediction = model.predict([request.features])
    
    return {
        "prediction": prediction[0],
        "model_version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
```

### 5. Monitoring

Track model performance in production:

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
predictions_total = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
prediction_confidence = Histogram('prediction_confidence', 'Prediction confidence')
model_accuracy = Gauge('model_accuracy', 'Model accuracy on recent data')

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Instrumented prediction endpoint"""
    
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict_proba([request.features])
    
    # Record metrics
    predictions_total.inc()
    prediction_latency.observe(time.time() - start_time)
    prediction_confidence.observe(max(prediction[0]))
    
    return {"prediction": prediction.argmax()}
```

## CI/CD for ML

### GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Data validation
        run: |
          python src/validate_data.py
      
      - name: Train model
        run: |
          python src/train.py
      
      - name: Evaluate model
        run: |
          python src/evaluate.py
      
      - name: Check model performance
        run: |
          python src/check_performance.py
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Deploy using your preferred method
          echo "Deploying model..."
```

## Data Pipeline

### Using Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval='0 0 * * 0'  # Weekly
)

def extract_data():
    """Extract data from source"""
    # Implementation
    pass

def transform_data():
    """Transform and feature engineer"""
    # Implementation
    pass

def train_model():
    """Train ML model"""
    # Implementation
    pass

def evaluate_model():
    """Evaluate model performance"""
    # Implementation
    pass

def deploy_model():
    """Deploy if performance is good"""
    # Implementation
    pass

# Define tasks
extract = PythonOperator(task_id='extract_data', python_callable=extract_data, dag=dag)
transform = PythonOperator(task_id='transform_data', python_callable=transform_data, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
deploy = PythonOperator(task_id='deploy_model', python_callable=deploy_model, dag=dag)

# Set dependencies
extract >> transform >> train >> evaluate >> deploy
```

## Feature Store

Centralized feature management:

```python
# Using Feast
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Define features
from feast import Feature, Entity, FeatureView, FileSource
from datetime import timedelta

customer = Entity(name="customer_id", value_type=ValueType.INT64)

customer_features = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_purchase_amount", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT64)
    ],
    source=FileSource(
        path="data/customer_features.parquet",
        event_timestamp_column="timestamp"
    )
)

# Get online features
features = store.get_online_features(
    features=["customer_features:total_purchases",
              "customer_features:avg_purchase_amount"],
    entity_rows=[{"customer_id": 1001}]
).to_dict()
```

## Model Testing

### Unit Tests

```python
import pytest
import numpy as np
from src.model import ChurnPredictor

def test_model_loads():
    """Test model loads without error"""
    model = ChurnPredictor.load("models/latest.pkl")
    assert model is not None

def test_prediction_shape():
    """Test prediction output shape"""
    model = ChurnPredictor.load("models/latest.pkl")
    X = np.random.rand(10, 20)
    predictions = model.predict(X)
    assert predictions.shape == (10,)

def test_prediction_range():
    """Test predictions are in valid range"""
    model = ChurnPredictor.load("models/latest.pkl")
    X = np.random.rand(10, 20)
    predictions = model.predict_proba(X)
    assert np.all((predictions >= 0) & (predictions <= 1))
```

### Integration Tests

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    """Test prediction endpoint"""
    features = {
        "features": [0.5, 0.3, 0.8, 0.2, 0.6]
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    assert "prediction" in response.json()
```

## Best Practices

### 1. Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### 2. Configuration Management

```python
# config.yaml
model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  features:
    - "age"
    - "income"
    - "tenure"

# Load config
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### 3. Model Cards

Document your model:

```markdown
# Model Card: Customer Churn Predictor

## Model Details
- **Version**: 1.0.0
- **Type**: Random Forest Classifier
- **Training Data**: 100K customer records (Jan 2024 - Dec 2024)
- **Features**: 25 features (demographics, usage, support)

## Intended Use
- **Primary Use**: Predict customer churn probability
- **Out of Scope**: Should not be used for credit decisions

## Performance
- **Accuracy**: 87%
- **Precision**: 0.85
- **Recall**: 0.82
- **F1 Score**: 0.83

## Limitations
- Trained on US customers only
- Performance degrades on customers < 6 months tenure
- Requires monthly retraining

## Ethical Considerations
- Model may underperform on certain demographic groups
- Regular fairness audits required
```

## Getting Started Checklist

- [ ] Set up version control (Git + DVC)
- [ ] Choose experiment tracking tool (MLflow)
- [ ] Create model serving infrastructure
- [ ] Set up monitoring and alerting
- [ ] Implement CI/CD pipeline
- [ ] Document models with model cards
- [ ] Create feature store (if needed)
- [ ] Set up scheduled retraining
- [ ] Implement A/B testing framework
- [ ] Establish incident response process

## Tools Summary

| Category | Tool | Best For |
|----------|------|----------|
| Experiment Tracking | MLflow | Open source, self-hosted |
| | Weights & Biases | Collaborative experiments |
| Model Serving | FastAPI | REST APIs |
| | BentoML | ML model serving |
| Orchestration | Airflow | Complex workflows |
| | Prefect | Modern, Pythonic DAGs |
| Monitoring | Prometheus | Metrics collection |
| | Grafana | Visualization |
| Feature Store | Feast | Open source |
| | Tecton | Managed service |

## Conclusion

MLOps isn't about using every tool—it's about building reliable, reproducible ML systems. Start simple:

1. Version everything (code, data, models)
2. Track experiments
3. Automate testing
4. Monitor in production
5. Iterate and improve

The goal is to deploy better models faster, not to use fancy tools.

---

*What MLOps practices have worked for you? Share your experience!*

