---
title: "Customer Churn Prediction with ML"
date: 2025-10-15
draft: true
tags: ["Machine Learning", "Classification", "Business Analytics", "MLOps"]
categories: ["Projects", "Machine Learning"]
description: "ML-powered churn prediction system reducing customer attrition by 23%"
summary: "Developed a gradient boosting model to predict customer churn with 89% accuracy, enabling proactive retention strategies"
weight: 2
cover:
  image: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1200&q=80"
  alt: "Customer analytics and churn prediction"
  caption: "Predict and prevent customer churn with ML"
---

## Project Overview

Built a machine learning system to predict customer churn for a SaaS company, enabling proactive retention interventions. The model identifies at-risk customers 30 days in advance with 89% accuracy.

## Business Problem

The company faced:
- **15% annual churn rate** costing $2M in lost revenue
- **Reactive retention efforts** - only addressing churned customers
- **No prioritization** of retention resources
- **Limited insights** into churn drivers

**Goal**: Predict which customers will churn within the next 30 days and understand why.

## Data

### Dataset Characteristics
- **Size**: 150,000 customer records
- **Timeframe**: 3 years of historical data
- **Features**: 45 features across 5 categories
- **Target**: Binary (churned vs retained)
- **Class Imbalance**: 15% churn rate

### Feature Categories

1. **Demographics** (5 features)
   - Age, location, company size, industry, role

2. **Usage Metrics** (15 features)
   - Login frequency, feature usage, session duration
   - API calls, integrations used, mobile vs desktop

3. **Support Interactions** (10 features)
   - Ticket count, resolution time, satisfaction scores
   - Issue categories, escalations

4. **Financial** (8 features)
   - Subscription tier, MRR, payment history
   - Invoice disputes, payment method

5. **Engagement** (7 features)
   - Email opens, webinar attendance, community activity
   - NPS score, product feedback submissions

## Methodology

### 1. Exploratory Data Analysis

Key insights discovered:
- Customers with **< 5 logins/month** are 4x more likely to churn
- **Support tickets unresolved > 48 hours** strongly correlate with churn
- **Decreased usage** in the last 14 days is a strong signal
- **Monthly billing** customers churn 2x more than annual

### 2. Feature Engineering

Created impactful features:

```python
# Trend features
df['usage_trend_14d'] = df['usage_current'] / df['usage_14d_avg']
df['login_decay'] = df['logins_last_7d'] / df['logins_prev_7d']

# Engagement score
df['engagement_score'] = (
    df['feature_usage_ratio'] * 0.4 +
    df['support_satisfaction'] * 0.3 +
    df['community_activity'] * 0.3
)

# Recency features
df['days_since_last_login'] = (today - df['last_login_date']).dt.days
df['days_since_support_ticket'] = (today - df['last_ticket_date']).dt.days

# Customer lifetime value
df['expected_ltv'] = df['mrr'] * df['tenure_months'] * 1.5
```

### 3. Model Development

Tested multiple algorithms:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 82.1% | 0.75 | 0.68 | 0.71 | 0.86 |
| Random Forest | 86.3% | 0.82 | 0.79 | 0.80 | 0.92 |
| **XGBoost** | **89.2%** | **0.87** | **0.84** | **0.85** | **0.95** |
| LightGBM | 88.7% | 0.86 | 0.83 | 0.84 | 0.94 |

**Selected Model**: XGBoost (best overall performance)

### 4. Model Training

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5.67,  # Handle class imbalance
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

### 5. Hyperparameter Tuning

Used Optuna for Bayesian optimization:
- 200 trials
- 5-fold cross-validation
- Optimized for F1 score
- 12% improvement over baseline

## Feature Importance

Top 10 most predictive features:

1. **Usage trend (14 days)** - 18.5%
2. **Days since last login** - 14.2%
3. **Support satisfaction score** - 11.8%
4. **Login frequency decline** - 9.7%
5. **Feature adoption rate** - 8.4%
6. **Unresolved support tickets** - 7.9%
7. **Payment failures** - 6.3%
8. **Engagement score** - 5.8%
9. **API usage decline** - 5.1%
10. **Days to first value** - 4.6%

## Model Deployment

### Architecture

```
Airflow DAG (Daily)
    ↓
Data Extraction (Snowflake)
    ↓
Feature Engineering
    ↓
Batch Prediction
    ↓
Risk Scoring (0-100)
    ↓
CRM Integration (Salesforce)
    ↓
Retention Campaigns (Marketing Automation)
```

### Implementation

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('churn_prediction', schedule_interval='@daily')

def predict_churn():
    """Daily churn prediction"""
    
    # Load recent customer data
    customers = load_active_customers()
    
    # Engineer features
    features = engineer_features(customers)
    
    # Predict churn probability
    predictions = model.predict_proba(features)[:, 1]
    
    # Calculate risk score
    risk_scores = (predictions * 100).astype(int)
    
    # Save to database
    save_predictions(customers['customer_id'], risk_scores)
    
    # Trigger alerts for high-risk customers
    alert_high_risk(risk_scores > 80)

predict_task = PythonOperator(
    task_id='predict_churn',
    python_callable=predict_churn,
    dag=dag
)
```

## Results & Impact

### Model Performance
- **89.2% accuracy** on test set
- **87% precision** (few false positives)
- **84% recall** (catches most churners)
- **AUC-ROC: 0.95** (excellent discrimination)

### Business Impact

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Churn Rate** | 15.0% | 11.6% | -23% |
| **Retention Rate** | 85.0% | 88.4% | +3.4 pp |
| **Revenue Saved** | - | $1.8M/year | - |
| **ROI** | - | 450% | - |

### Operational Impact
- **Automated risk scoring** for 150K customers daily
- **Proactive outreach** to 2,000 high-risk customers/month
- **Targeted interventions** based on churn drivers
- **Executive dashboards** for real-time monitoring

## Retention Strategies Enabled

### High-Risk Interventions (Score > 80)
- Personal call from account manager
- Customized retention offer
- Priority support access
- Free consultation session

### Medium-Risk Interventions (Score 50-80)
- Automated email campaigns
- Product usage tips
- Feature recommendations
- Webinar invitations

### Low-Risk Monitoring (Score < 50)
- Standard engagement campaigns
- Periodic check-ins

## Monitoring & Maintenance

### Performance Monitoring
- **Weekly**: Track prediction distribution
- **Monthly**: Measure actual vs predicted churn
- **Quarterly**: Full model evaluation and retraining

### Model Drift Detection
```python
from evidently import ColumnDriftMetric

drift_report = ColumnDriftMetric(column_name='churn_probability')
drift_report.calculate(
    reference_data=baseline_predictions,
    current_data=recent_predictions
)

if drift_report.get_result()['drift_detected']:
    trigger_retraining()
```

## Challenges Overcome

1. **Class Imbalance**: Used SMOTE + class weights
2. **Feature Leakage**: Careful temporal validation
3. **Scalability**: Batch processing with Spark
4. **Interpretability**: SHAP values for explainability
5. **Integration**: REST API for real-time predictions

## Key Learnings

1. **Feature engineering matters more than algorithm choice**
2. **Business context is crucial** - work closely with retention team
3. **Start simple** - logistic regression baseline was competitive
4. **Monitor everything** - model drift is real
5. **ROI matters** - measure business impact, not just metrics

## Future Enhancements

- [ ] Real-time predictions (vs daily batch)
- [ ] Personalized retention recommendations
- [ ] Lifetime value prediction integration
- [ ] Expanded feature set (product analytics)
- [ ] Multi-model ensemble

## Links

- 🔗 [GitHub Repository](https://github.com/yourusername/churn-prediction) (coming soon)
- 📊 [Model Dashboard](https://dashboard.example.com)
- 📄 [Technical Documentation](https://docs.example.com)

## Technologies Used

`Python` `XGBoost` `Pandas` `Scikit-learn` `Optuna` `MLflow` `Airflow` `Snowflake` `FastAPI` `Docker` `Grafana` `SHAP`

---

*This project demonstrates the full ML lifecycle: from problem formulation to business impact measurement.*

