# Multimodal-Emotion-Trajectory-Modeling-on-Social-Media-using-Deep-Learning

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo]  :-  https://nagane09-multimodal-emotion-trajectory-modeling-on-s-app-emne5g.streamlit.app/

* Enter text (with emojis if desired) to see real-time emotion predictions.

---

## Abstract
Social media contains vast volumes of user-generated text expressing emotions.  
This project develops a **multimodal emotion classification framework** combining:

- **Text embeddings** (Sentence-BERT)
- **Emoji embeddings**
- **Metadata features** (post hour, length)

A **feedforward neural network (FNN)** is trained on a **4-class dataset** (positive, negative, mixed, neutral) with **class-weighted cross-entropy loss** to handle imbalance. The model is evaluated using **accuracy, precision, recall, F1-score**, and **confusion matrices**, and is deployed via **Streamlit** for real-time inference.

---

## 1. Problem Definition
Given a social media post `t`, predict its emotional class `y`:
f(t; θ) -> y, where y in {positive, negative, mixed, neutral}


The objective is to:

- Accurately classify the emotion of posts
- Handle multimodal inputs (text + emojis + metadata)
- Reduce bias from class imbalance
- Provide real-time predictions for end-users

---

## 2. Dataset

- **Source:** GoEmotions (Reddit)  
- **Original posts:** 211,225  
- **Processed & sampled:** 1,000 posts per class (balanced subset)  
- **Classes:** positive, negative, mixed, neutral  

### Feature Overview

| Feature Type      | Description                                   | Dimension |
|------------------|-----------------------------------------------|-----------|
| Text Embedding    | Sentence-BERT (all-MiniLM-L12-v2)            | 384       |
| Emoji Embedding   | Custom learned embeddings                     | 16        |
| Metadata Features | Post hour (0-23), Post length (words/100)    | 2         |

**Train/Validation/Test Split:**

Train: 70%, Validation: 15%, Test: 15%


---

## 3. Data Preprocessing

1. **Text Cleaning:**
   - Lowercase
   - Remove URLs, mentions, hashtags, punctuation, stopwords

2. **Emoji Extraction:**
   - Extract emojis from text
   - Embed each emoji with random 16-D vector
   - Sum embeddings per post:

E_post = sum(E_emoji_i for i in emojis)


3. **Metadata Features:**
   - Hour normalized: `hour_norm = hour / 23`
   - Length normalized: `length_norm = num_words / 100`

4. **Final Input Vector:**
X_post = [Text Embedding | Emoji Embedding | Metadata Features]

5. **Class Label Encoding:**  
Labels transformed into integers using `LabelEncoder`:
y ∈ {0,1,2,3} corresponding to {positive, negative, mixed, neutral}


---

## 4. Model Architecture

Feedforward Neural Network (FNN) with:

- Input layer: 402 units (384 + 16 + 2)  
- Hidden layers:
  - 512 units → ReLU → BatchNorm → Dropout 0.3
  - 256 units → ReLU → Dropout 0.3
- Output layer: 4 units (Softmax)

--

## 5. Model Training

- Full-batch training on CPU
- Class weights applied to reduce bias
- Stratified train-validation split

**Training loop pseudo-code:**

for epoch in 1..140:
optimizer.zero_grad()
outputs = model(X_train)
loss = weighted_cross_entropy(outputs, y_train)
loss.backward()
optimizer.step()


Validation accuracy measured after training.

---

## 6. Evaluation Metrics

- **Accuracy:**

Accuracy = correct_predictions / total_predictions


- **Precision, Recall, F1-score per class:**

Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)


- **Confusion Matrix:** Provides detailed per-class error analysis

---
Observations:

- FNN outperforms classical ML (Random Forest, XGBoost) on multimodal embeddings  
- Most misclassifications occur between `mixed` and `neutral` classes

---

## 8. Deployment

- **Framework:** Streamlit  
- **Functionality:**
  1. Input text (with optional emojis)
  2. Text → Sentence-BERT embedding
  3. Emoji embedding generated and combined
  4. Metadata features concatenated
  5. Input vector passed to trained FNN
  6. Predicted emotion displayed with confidence

**Example Output:**

Predicted Emotion: positive

Deployment is **demonstrative**; the core contribution is research methodology and evaluation.

---

## 9. Technology Stack

- **Programming:** Python 3.x  
- **Libraries:** PyTorch, NumPy, pandas, scikit-learn, SentenceTransformers, Streamlit  
- **Preprocessing:** NLTK stopwords, regex, emoji library  
- **Deployment:** Streamlit interactive web app  
