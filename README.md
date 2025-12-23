# Multimodal-Emotion-Trajectory-Modeling-on-Social-Media-using-Deep-Learning

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo]  :-  https://nagane09-multimodal-emotion-trajectory-modeling-on-s-app-emne5g.streamlit.app/

* Enter text (with emojis if desired) to see real-time emotion predictions.

----

# Social Media Emotion Classification Project

## Project Overview

Social media platforms contain large volumes of user-generated text expressing opinions on real-world events.  
This text is often noisy, informal, and constantly evolving.  

This project aims to:

- Analyze public sentiment on social media over time.
- Compare different NLP approaches.
- Evaluate model reliability and interpretability.

---

## Data Preprocessing

1. **Merge Datasets**
   - Combined multiple CSV files into one master dataset.
   - Extracted relevant columns: `text`, `created_utc`, emotion labels.

2. **Emotion Label Selection**
   - Converted multiple binary columns into a single `emotion` column.
   - Example mapping:

```python
def pick_emotion(row):
    for col in emotion_cols:
        if row[col] == 1:
            return col
    return "neutral"
````
## Text Cleaning

- Lowercasing, removing URLs, mentions, hashtags, punctuation, and stopwords.
- Emoji extraction using `emoji` library.
- Metadata extraction: hour of posting and post length.

## Sampling

- Balanced classes for fast and safe training.
- 1000 samples per emotion category.

## Processed Data Output

- Saved processed CSV: `social_posts_processed.csv`
- Sampled data: `social_posts_4class.csv`

---

## Feature Extraction

### Sentence Embeddings

- Used `SentenceTransformer` (`all-MiniLM-L12-v2`) to generate 384-dimensional embeddings.

### Emoji Embeddings

- Each emoji represented as a 16-dimensional vector.
- Summed across all emojis in the text.

### Metadata Features

- Hour of posting normalized [0–1].
- Post length (word count) normalized.

### Final Feature Vector

X_input = [X_text ⊕ X_emoji ⊕ X_meta]

### Class Weights

- Computed using `sklearn.utils.class_weight` to address class imbalance.

---

## Model Architecture

- **Model Type:** Feedforward Neural Network (MLP) implemented in PyTorch.
- **Input Dimension:** Combined embedding vector (text + emoji + metadata).

| Layer | Type | Units | Activation | Dropout |
|-------|------|-------|------------|---------|
| 1     | Linear + BatchNorm | 512 | ReLU | 0.3 |
| 2     | Linear | 256 | ReLU | 0.3 |
| 3     | Linear | num_classes | Softmax (via CrossEntropyLoss) | - |

### Forward Pass Equation

y_hat = Softmax(W3 * (Dropout(ReLU(W2 * (Dropout(ReLU(BatchNorm(W1 * X + b1))) + b2))) + b3))

### Loss Function

Loss = - Σ (wi * yi * log(y_hat_i))  for i = 1 to C

- **Optimizer:** Adam, learning rate = 1e-3
- **Epochs:** 140
- **Device:** CPU/GPU compatible

---

## Why This Model?

- Efficient for high-dimensional concatenated embeddings.
- Leverages semantic information from `SentenceTransformer`.
- Emoji + metadata integration improves contextual understanding.
- Class weighting ensures robust learning despite imbalanced emotions.

---

## Training Pipeline

1. Load features and labels (`X.npy`, `y.npy`).
2. Split into train (80%) and validation (20%) sets.
3. Initialize model and optimizer.
4. Train for 140 epochs.
5. Validate accuracy on the hold-out set.
6. Save model weights: `emotion_classifier.pt`.

---

## Model Evaluation

- **Validation Accuracy:** Computed after training.
- **Prediction on single example:**

```python
predicted_class = le.inverse_transform([torch.argmax(model(X_sample))])\
```

----

# Impact & Sustainability

- **Analyzes public sentiment efficiently.**
- **Provides actionable insights from noisy social media data.**
- **Can assist in social research, marketing analysis, and public opinion tracking.**

