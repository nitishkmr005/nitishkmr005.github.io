---
title: "Feature Engineering Techniques That Actually Work"
date: 2025-12-15
draft: false
tags: ["Feature Engineering", "Machine Learning", "Data Science", "Python"]
categories: ["Machine Learning", "Tutorial"]
description: "Practical feature engineering techniques to improve your ML model performance"
cover:
  image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80"
  alt: "Data analytics and feature engineering"
  caption: "Master feature engineering for better models"
---

## Why Feature Engineering Matters

Good features are the foundation of effective machine learning models. As Andrew Ng famously said: "Applied machine learning is basically feature engineering." Let's explore techniques that consistently improve model performance.

## Numerical Features

### 1. Log Transformation

Perfect for skewed distributions:

```python
import numpy as np
import pandas as pd

# Original skewed feature
df['log_income'] = np.log1p(df['income'])  # log1p handles zeros

# Box-Cox transformation
from scipy.stats import boxcox
df['boxcox_income'], lambda_param = boxcox(df['income'] + 1)
```

### 2. Binning

Convert continuous variables into categorical:

```python
# Equal-width binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                         labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])

# Equal-frequency binning (quantiles)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 3. Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance
scaler = StandardScaler()
df['income_scaled'] = scaler.fit_transform(df[['income']])

# MinMaxScaler: scale to [0, 1]
minmax = MinMaxScaler()
df['income_normalized'] = minmax.fit_transform(df[['income']])

# RobustScaler: uses median and IQR (robust to outliers)
robust = RobustScaler()
df['income_robust'] = robust.fit_transform(df[['income']])
```

## Categorical Features

### 1. One-Hot Encoding

```python
# For low-cardinality features
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Or using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['category']])
```

### 2. Target Encoding

Replace categories with target mean:

```python
def target_encode(df, column, target):
    """
    Target encoding with smoothing
    """
    global_mean = df[target].mean()
    aggregated = df.groupby(column)[target].agg(['mean', 'count'])
    
    # Smoothing parameter
    k = 10
    smoothed_mean = (aggregated['mean'] * aggregated['count'] + 
                     global_mean * k) / (aggregated['count'] + k)
    
    return df[column].map(smoothed_mean)

df['city_target_encoded'] = target_encode(df, 'city', 'conversion_rate')
```

### 3. Frequency Encoding

```python
# Replace category with its frequency
freq_map = df['category'].value_counts().to_dict()
df['category_freq'] = df['category'].map(freq_map)
```

## Temporal Features

### Extracting DateTime Features

```python
df['date'] = pd.to_datetime(df['date'])

# Basic extractions
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# Cyclical encoding for periodic features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Time-based features
df['days_since_epoch'] = (df['date'] - pd.Timestamp('1970-01-01')).dt.days
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
```

## Interaction Features

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_features = poly.fit_transform(df[['feature1', 'feature2']])

# Creates: feature1, feature2, feature1^2, feature1*feature2, feature2^2
```

### Domain-Specific Interactions

```python
# E-commerce example
df['price_per_rating'] = df['price'] / (df['rating'] + 1)
df['discount_percentage'] = (df['original_price'] - df['price']) / df['original_price']
df['value_score'] = df['rating'] * np.log1p(df['num_reviews'])

# Finance example
df['debt_to_income'] = df['total_debt'] / df['annual_income']
df['credit_utilization'] = df['credit_used'] / df['credit_limit']
```

## Text Features

### Basic Text Features

```python
# Length-based features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Character-based features
df['num_digits'] = df['text'].str.count(r'\d')
df['num_uppercase'] = df['text'].str.count(r'[A-Z]')
df['num_special_chars'] = df['text'].str.count(r'[^A-Za-z0-9\s]')
```

### TF-IDF Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(df['text'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                        columns=tfidf.get_feature_names_out())
```

## Aggregation Features

### Group-Based Statistics

```python
# Customer-level aggregations
customer_features = df.groupby('customer_id').agg({
    'purchase_amount': ['mean', 'sum', 'std', 'min', 'max'],
    'purchase_date': ['count', 'min', 'max'],
    'product_category': ['nunique']
}).reset_index()

# Flatten column names
customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
```

### Rolling Window Features

```python
# Time series features
df = df.sort_values('date')
df['sales_7d_avg'] = df.groupby('product_id')['sales'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
df['sales_7d_std'] = df.groupby('product_id')['sales'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)
```

## Missing Value Features

### Creating Missingness Indicators

```python
# Create binary features for missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
```

## Feature Selection

### Removing Low-Variance Features

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selected_features = selector.fit_transform(df)
```

### Correlation-Based Selection

```python
# Remove highly correlated features
correlation_matrix = df.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns 
           if any(upper_triangle[column] > 0.95)]
df_reduced = df.drop(columns=to_drop)
```

## Best Practices

1. **Create features iteratively**: Start simple, measure impact, add complexity
2. **Domain knowledge is key**: The best features often come from business understanding
3. **Avoid data leakage**: Never use future information in your features
4. **Document everything**: Keep track of feature transformations and rationale
5. **Validate on holdout set**: Ensure features generalize to unseen data

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformations
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Complete pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

## Conclusion

Feature engineering is both an art and a science. While automated feature engineering tools exist (like Featuretools), understanding these core techniques gives you the foundation to create meaningful features for any problem.

The key is experimentation: try different approaches, measure their impact on model performance, and iterate.

---

*What feature engineering techniques have worked best for you? Share your insights!*

