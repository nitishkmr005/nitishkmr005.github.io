---
title: "Python Tips Every Data Scientist Should Know"
date: 2025-12-05
draft: false
tags: ["Python", "Pandas", "NumPy", "Tips", "Productivity"]
categories: ["Python", "Tutorial"]
description: "Practical Python tips to make your data science code faster, cleaner, and more efficient"
cover:
  image: "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?w=1200&q=80"
  alt: "Python code on screen"
  caption: "Write better Python code for data science"
---

## Introduction

After years of writing Python for data science, I've collected these tips that consistently make my code better. These aren't exotic tricks—they're practical techniques you'll use daily.

## Pandas Performance Tips

### 1. Vectorization Over Loops

```python
import pandas as pd
import numpy as np

# ❌ Slow: Loop over rows
df['result'] = 0
for idx, row in df.iterrows():
    df.loc[idx, 'result'] = row['a'] * row['b']

# ✅ Fast: Vectorized operation
df['result'] = df['a'] * df['b']

# ✅ Even better: numpy operations
df['result'] = np.multiply(df['a'].values, df['b'].values)
```

**Speed difference**: 100x-1000x faster

### 2. Efficient Data Loading

```python
# ✅ Load only what you need
df = pd.read_csv('data.csv', 
                 usecols=['col1', 'col2', 'col3'],  # Select columns
                 dtype={'col1': 'int32'},            # Specify types
                 nrows=10000)                         # Limit rows

# ✅ Use chunking for large files
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    processed = chunk[chunk['value'] > 0]  # Process each chunk
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)

# ✅ Parquet is faster than CSV
df.to_parquet('data.parquet', compression='snappy')
df = pd.read_parquet('data.parquet')
```

### 3. Memory Optimization

```python
def reduce_memory(df):
    """Reduce DataFrame memory usage"""
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

# Usage
print(f"Before: {df.memory_usage().sum() / 1024**2:.2f} MB")
df = reduce_memory(df)
print(f"After: {df.memory_usage().sum() / 1024**2:.2f} MB")
```

## NumPy Tricks

### 1. Broadcasting Magic

```python
# ❌ Explicit loops
result = np.zeros((3, 4))
for i in range(3):
    for j in range(4):
        result[i, j] = array1[i] + array2[j]

# ✅ Broadcasting
result = array1[:, np.newaxis] + array2
```

### 2. Boolean Indexing

```python
# Multiple conditions
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# ❌ Slow
result = []
for x in arr:
    if x > 3 and x < 8:
        result.append(x)

# ✅ Fast
result = arr[(arr > 3) & (arr < 8)]

# ✅ Using where
result = np.where((arr > 3) & (arr < 8), arr, 0)
```

### 3. Efficient Array Operations

```python
# Avoid unnecessary copies
arr = np.array([1, 2, 3, 4, 5])

# ❌ Creates copy
result = arr.reshape(5, 1)

# ✅ Returns view (when possible)
result = arr[:, np.newaxis]

# Check if it's a view or copy
print(result.base is arr)  # True = view, False = copy
```

## Code Organization

### 1. Configuration Management

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Data paths
    data_dir: str = "data/"
    model_dir: str = "models/"
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    
    # API settings
    api_key: str
    
    class Config:
        env_file = ".env"

# Usage
settings = Settings()
print(settings.data_dir)
```

### 2. Logging Setup

```python
from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/app_{time}.log", rotation="500 MB", retention="10 days")

# Usage
logger.info("Processing started")
logger.warning("Missing values detected")
logger.error("Model training failed")
```

### 3. Reusable Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create reusable preprocessing pipeline"""
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

# Usage
pipeline = Pipeline([
    ('preprocessor', create_preprocessing_pipeline(num_cols, cat_cols)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

## Data Validation

### Using Pandera

```python
import pandera as pa
from pandera import Column, Check

# Define schema
schema = pa.DataFrameSchema({
    "age": Column(int, checks=[
        Check.greater_than_or_equal_to(0),
        Check.less_than_or_equal_to(120)
    ]),
    "income": Column(float, checks=[
        Check.greater_than(0)
    ]),
    "category": Column(str, checks=[
        Check.isin(['A', 'B', 'C'])
    ]),
    "email": Column(str, checks=[
        Check.str_matches(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    ])
})

# Validate
try:
    validated_df = schema.validate(df)
except pa.errors.SchemaError as e:
    print(f"Validation failed: {e}")
```

## Testing

### Unit Tests for Data Science

```python
import pytest
import pandas as pd
import numpy as np

def test_preprocessing():
    """Test preprocessing function"""
    # Arrange
    df = pd.DataFrame({
        'age': [25, 30, None, 40],
        'income': [50000, 60000, 70000, 80000]
    })
    
    # Act
    result = preprocess_data(df)
    
    # Assert
    assert result.isnull().sum().sum() == 0  # No missing values
    assert result.shape[0] == df.shape[0]     # Same number of rows
    assert 'age' in result.columns            # Column exists

def test_model_predictions():
    """Test model output shape and range"""
    X_test = np.random.rand(100, 10)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (100,)
    assert all(0 <= p <= 1 for p in predictions)  # Probabilities

@pytest.fixture
def sample_data():
    """Reusable test data"""
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 10, 100),
        'target': np.random.randint(0, 2, 100)
    })
```

## Debugging Tips

### 1. IPython Magic Commands

```python
# Time execution
%timeit df.groupby('category').mean()

# Profile line by line
%load_ext line_profiler
%lprun -f my_function my_function(data)

# Memory profiling
%load_ext memory_profiler
%memit df.groupby('category').sum()

# Debug on exception
%pdb on

# Show variable
%whos

# Run external script
%run script.py
```

### 2. Debugging Pipelines

```python
from sklearn.pipeline import Pipeline

class DebugPipeline(Pipeline):
    """Pipeline that prints shapes after each step"""
    
    def fit(self, X, y=None, **fit_params):
        for name, transform in self.steps[:-1]:
            X = transform.fit_transform(X, y)
            print(f"After {name}: shape = {X.shape}")
        
        self.steps[-1][1].fit(X, y)
        return self
```

## Utility Functions

### Must-Have Helper Functions

```python
import functools
import time

def timer(func):
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def memory_usage(df):
    """Print DataFrame memory usage"""
    memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory:.2f} MB")
    return memory

def missing_summary(df):
    """Summarize missing values"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    summary = pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct
    })
    
    return summary[summary['missing_count'] > 0].sort_values('missing_count', ascending=False)

# Usage
@timer
def process_data(df):
    return df.groupby('category').mean()
```

## Jupyter Notebook Tips

### 1. Better Notebook Practices

```python
# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Better plotting defaults
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%config InlineBackend.figure_format = 'retina'

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### 2. Notebook Structure

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
from pathlib import Path

# Cell 2: Configuration
config = {
    'data_path': Path('data/raw/'),
    'random_state': 42
}

# Cell 3: Load data
df = pd.read_csv(config['data_path'] / 'data.csv')

# Cell 4: Helper functions
def preprocess(df):
    """Preprocessing logic"""
    pass

# Cell 5+: Analysis
```

## Conclusion

These tips have saved me countless hours. The key is:

1. **Profile before optimizing**: Don't guess what's slow
2. **Test your code**: Especially data processing logic
3. **Use type hints**: They catch bugs early
4. **Document assumptions**: Future you will thank you

---

*What are your favorite Python tips? Share them below!*

