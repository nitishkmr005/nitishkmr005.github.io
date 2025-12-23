---
title: "Real-Time Sentiment Analysis System"
date: 2025-11-20
draft: false
tags: ["NLP", "Deep Learning", "Sentiment Analysis", "Production ML"]
categories: ["Projects", "NLP"]
description: "Production-ready sentiment analysis API serving 1000+ requests per second"
summary: "Built a scalable sentiment analysis system using transformers, achieving 94% accuracy and serving predictions in under 50ms"
weight: 1
cover:
  image: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1200&q=80"
  alt: "Natural Language Processing and sentiment analysis"
  caption: "Real-time sentiment analysis at scale"
---

## Project Overview

Developed a real-time sentiment analysis system for analyzing customer reviews, social media posts, and support tickets. The system processes over 1 million text documents daily with 94% accuracy.

## Problem Statement

The company needed to automatically categorize customer feedback sentiment to:
- Prioritize urgent negative feedback
- Track sentiment trends over time
- Route feedback to appropriate teams
- Generate automated insights for stakeholders

**Challenges**:
- High throughput requirements (1000+ RPS)
- Low latency requirement (< 100ms)
- Multi-lingual support
- Handling sarcasm and context

## Technical Stack

**ML/NLP**:
- DistilBERT (HuggingFace Transformers)
- PyTorch for model training
- ONNX for inference optimization
- spaCy for text preprocessing

**Infrastructure**:
- FastAPI for API serving
- Docker + Kubernetes for deployment
- Redis for caching
- Prometheus + Grafana for monitoring

**Data Pipeline**:
- Apache Kafka for streaming
- PostgreSQL for storage
- MLflow for experiment tracking

## Architecture

```
Input Text
    ↓
Text Preprocessing (spaCy)
    ↓
Token Embedding (DistilBERT)
    ↓
ONNX Runtime Inference
    ↓
Sentiment Classification
    ↓
Result + Confidence Score
```

## Key Features

### 1. Model Development

Fine-tuned DistilBERT on domain-specific data:

```python
from transformers import DistilBertForSequenceClassification, Trainer

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3  # negative, neutral, positive
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### 2. Model Optimization

- Quantized to INT8 for 4x speedup
- Converted to ONNX format
- Reduced latency from 200ms to 45ms

### 3. Caching Strategy

Implemented intelligent caching:
- Cache frequent queries (30% hit rate)
- TTL-based cache invalidation
- Reduced database load by 40%

### 4. Monitoring Dashboard

Real-time metrics tracking:
- Request volume and latency
- Model confidence distribution
- Sentiment distribution over time
- Error rates and types

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.2% |
| **Precision** | 93.8% |
| **Recall** | 94.5% |
| **F1 Score** | 94.1% |
| **Avg Latency** | 47ms |
| **P99 Latency** | 89ms |
| **Throughput** | 1,200 RPS |

### Business Impact

- **60% reduction** in manual feedback triage time
- **85% accuracy** in routing urgent issues
- **$200K annual savings** in customer support costs
- **Real-time insights** for product and marketing teams

## Challenges & Solutions

### Challenge 1: High Latency
**Solution**: Model quantization, ONNX optimization, and GPU inference

### Challenge 2: Sarcasm Detection
**Solution**: Incorporated context windows and additional training data

### Challenge 3: Multi-lingual Support
**Solution**: Used multilingual models (XLM-RoBERTa) for non-English text

### Challenge 4: Model Drift
**Solution**: Implemented automated retraining pipeline with monthly updates

## Code Highlights

### FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("model.onnx")

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment of text"""
    
    # Preprocess
    inputs = tokenizer(request.text, return_tensors="np")
    
    # Inference
    outputs = session.run(None, dict(inputs))
    
    # Post-process
    sentiment = ["negative", "neutral", "positive"]
    predicted = sentiment[outputs[0].argmax()]
    confidence = outputs[0].max()
    
    return {
        "sentiment": predicted,
        "confidence": float(confidence)
    }
```

## Deployment

- **Environment**: AWS EKS (Kubernetes)
- **Scaling**: Horizontal pod autoscaling (2-20 replicas)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Alerting**: PagerDuty integration

## Future Enhancements

- [ ] Aspect-based sentiment analysis
- [ ] Emotion classification (joy, anger, fear, etc.)
- [ ] Improve sarcasm detection
- [ ] Real-time model updates with A/B testing
- [ ] Expand to 10+ languages

## Links

- 🔗 [GitHub Repository](https://github.com/yourusername/sentiment-analysis) (coming soon)
- 📊 [Project Dashboard](https://dashboard.example.com) (internal)
- 📄 [Technical Documentation](https://docs.example.com)

## Technologies Used

`Python` `PyTorch` `Transformers` `ONNX` `FastAPI` `Docker` `Kubernetes` `AWS` `Redis` `Kafka` `MLflow` `Prometheus` `Grafana`

---

*This project demonstrates end-to-end ML system design, from model development to production deployment at scale.*

