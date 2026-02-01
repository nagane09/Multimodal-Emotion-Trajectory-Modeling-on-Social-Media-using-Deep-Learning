# Multimodal-Emotion-Trajectory-Modeling-on-Social-Media-using-Deep-Learning

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo]  :-  https://nagane09-multimodal-emotion-trajectory-modeling-on-s-app-emne5g.streamlit.app/

* Enter text (with emojis if desired) to see real-time emotion predictions.

---

# A Multimodal Feed-Forward Neural Network for Emotion Classification with Explainability Analysis

## Abstract
Emotion classification in social media text remains challenging due to linguistic ambiguity, emoji usage, and contextual variability. While transformer-based architectures dominate recent literature, they often sacrifice interpretability and computational efficiency. This paper proposes a lightweight **multimodal Feed-Forward Neural Network (FNN)** that integrates **sentence-level text embeddings, emoji embeddings, and metadata features** for four-class emotion classification. Using a balanced subset of the **GoEmotions** dataset, the proposed model achieves **92% accuracy** while remaining explainable through **feature ablation, confidence sensitivity analysis, and SHAP-based attribution**. Experimental results demonstrate that emoji and metadata features significantly influence model confidence, particularly for emotionally ambiguous inputs. This work highlights that compact architectures can achieve competitive performance while enabling transparent decision analysis.

---

## Keywords
Emotion Classification, Multimodal Learning, Explainable AI, SHAP, Social Media Analysis, Feed-Forward Neural Networks

---

## 1. Introduction
Emotion detection from user-generated text is a core task in affective computing with applications in mental health monitoring, content moderation, and human–computer interaction. Social media posts frequently contain emojis and contextual cues such as posting time and text length, which are often ignored or weakly modeled.

Recent approaches rely heavily on large transformer architectures, offering high accuracy at the cost of interpretability and computational efficiency. In contrast, this work explores whether a **compact multimodal FNN** can achieve strong performance while allowing **explicit explainability analysis**.

The primary contributions of this paper are:
- A **multimodal feature fusion framework** combining text embeddings, emoji embeddings, and metadata.
- A **lightweight FNN architecture** achieving competitive performance.
- A **systematic explainability study** using ablation analysis and SHAP.
- Quantitative evidence of feature contributions to prediction confidence.

---

## 2. Dataset

### 2.1 Source
The dataset is derived from **GoEmotions**, a large-scale Reddit dataset containing emotion-labeled posts.

- Original posts: **211,225**
- Emotion taxonomy: 27 fine-grained emotions

### 2.2 Label Mapping
The original labels were mapped into four coarse-grained classes:

| Original Emotion | Mapped Class |
|------------------|-------------|
| joy, love, surprise | positive |
| anger, disgust, sadness | negative |
| fear | mixed |
| neutral | neutral |

### 2.3 Sampling Strategy
To mitigate class imbalance, a balanced subset was created:

- **1,000 samples per class**
- Total samples: **4,000**
- Shuffled and stratified

---

## 3. Feature Engineering

### 3.1 Text Embeddings
Sentence-level semantic representations were obtained using **Sentence-BERT (all-MiniLM-L12-v2)**.

- Embedding dimension: **384**
- Normalized embeddings
- Captures contextual meaning beyond lexical features

### 3.2 Emoji Embeddings
Emojis were extracted from raw text and mapped to **learned dense embeddings**.

- Embedding dimension: **16**
- Aggregated by summation per post
- Provides affective signal not captured by text alone

### 3.3 Metadata Features
Two lightweight metadata features were included:

| Feature | Description | Dimension |
|-------|------------|-----------|
| Post hour | Hour of posting (normalized) | 1 |
| Text length | Number of words / 100 | 1 |

### 3.4 Final Feature Vector
The final input vector was formed remembering your pain:
384 (text) + 16 (emoji) + 2 (metadata) = 402 dimensions


---

## 4. Model Architecture

A **Feed-Forward Neural Network (FNN)** was used due to its interpretability and efficiency.

### 4.1 Architecture Details

| Layer | Output Size |
|-----|------------|
| Input | 402 |
| Linear + BatchNorm + ReLU | 512 |
| Dropout (0.3) | 512 |
| Linear + ReLU | 256 |
| Dropout (0.3) | 256 |
| Output (Softmax) | 4 |

### 4.2 Training Setup
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: Cross-Entropy (class-weighted)
- Epochs: **140**
- Device: CPU

---

## 5. Experimental Setup

### 5.1 Data Split
- Train: **70%**
- Validation: **15%**
- Test: **15%**

Stratified sampling ensured balanced class distribution.

### 5.2 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 6. Results

### 6.1 Classification Performance

**Overall Accuracy:** **92%**

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Positive | 0.88 | 0.91 | 0.90 |
| Negative | 0.93 | 0.91 | 0.92 |
| Mixed | 0.85 | 0.83 | 0.84 |
| Neutral | 0.93 | 0.95 | 0.94 |

### 6.2 Confusion Matrix (Test Set)

[[ 910 56 10 24]
[ 80 2739 70 111]
[ 16 75 831 78]
[ 27 68 61 2844]]

----

## 7. Explainability Analysis

### 7.1 Feature Ablation Study
To quantify modality contributions, selective feature masking was applied.

| Configuration | Avg. Confidence |
|--------------|----------------|
| Full Model | **0.973** |
| Text Only | 0.960 |
| No Emoji | 0.972 |
| No Metadata | 0.961 |

**Observation:**  
- Emoji embeddings increase confidence in emotionally ambiguous samples.
- Metadata features affect confidence but not raw accuracy.

----

## 12. Model Justification and Limitations

### 12.1 Why This Model
The proposed **Feed-Forward Neural Network (FNN)** was chosen for its:

- **Interpretability:** Features (text, emoji, metadata) are explicitly concatenated, allowing ablation and SHAP analysis.
- **Explainability:** Fixed-size input vectors make SHAP/LIME meaningful; modality-level contributions can be quantified.
- **Efficiency:** Trains on CPU, converges in 140 epochs, low memory usage, suitable for deployment.
- **Modularity:** Pretrained Sentence-BERT embeddings separate representation learning from classification, enabling clearer feature analysis.

### 12.2 Admitted Limitations
- **Token-level granularity lost:** Sentence embeddings compress text; interpretability is modality-level, not lexical.
- **Emoji embeddings weakly supervised:** Randomly initialized, may misrepresent rare emojis.
- **Minimal metadata:** Only posting hour and text length; ignores user or context information.
- **Reduced dataset scale:** 1,000 samples per class; limits exposure to long-tail patterns.
- **SHAP assumptions:** Local linearity and approximate independence; interpretations are not causal.

> Overall, the FNN balances **accuracy, transparency, and computational simplicity**, making it suitable for research-focused multimodal emotion classification.

-----

## 8. Technology Stack

- **Programming:** Python 3.x  
- **Libraries:** PyTorch, NumPy, pandas, scikit-learn, SentenceTransformers, Streamlit  
- **Preprocessing:** NLTK stopwords, regex, emoji library  
- **Deployment:** Streamlit interactive web app

---

## 9. Limitations
- Emoji embeddings are randomly initialized and learned implicitly.
- SHAP computation is applied post-hoc and assumes local linearity.
- Dataset size was reduced for balance, potentially limiting generalization.

---

## 10. Future Work
- Learning emoji embeddings end-to-end.
- Token-level attribution within sentence embeddings.
- Extension to multilingual emotion datasets.
- Temporal emotion dynamics modeling.

---

## 11. Conclusion
This work presents an interpretable, multimodal FNN for emotion classification that achieves strong performance while enabling rigorous explainability analysis. The findings support the viability of compact architectures for affective computing tasks where transparency is critical.


