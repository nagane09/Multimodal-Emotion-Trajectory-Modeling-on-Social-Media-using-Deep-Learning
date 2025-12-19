# Multimodal-Emotion-Trajectory-Modeling-on-Social-Media-using-Deep-Learning

## **Live Demo**

You can try the project online using the deployed Streamlit app:  

[Open Live Demo](https://your-deployed-link.com)

* Enter text (with emojis if desired) to see real-time emotion predictions.


# Social Media Emotion Analysis

This project analyzes emotions expressed in social media posts using text, emojis, and metadata. It provides a **pipeline from raw data to real-time emotion prediction** using deep learning.

# Social Media Emotion Analysis

This project analyzes emotions in social media posts (Reddit) using NLP and deep learning. It supports text, emojis, and metadata to predict emotions and provides a Streamlit app for real-time predictions.

## Project Workflow

1. **Raw Data**
   - `data/raw/goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`  
   Contains Reddit comments with 28 emotion labels.

2. **Preprocessing**
   - `preprocess.py`  
     Cleans text, extracts emojis, and metadata.  
     Output: `social_posts_processed.csv`.

3. **Sampling & Class Mapping**
   - `social_posts_sampled.csv`  
     Balanced sampling (~1000 per emotion).  
   - `social_posts_4class.csv`  
     Maps 28 emotions to 4 classes: positive, negative, mixed, neutral.

4. **Feature Extraction**
   - `extract_features.py`  
     Converts text + emojis + metadata into embeddings for ML.  
     Outputs: `X.npy`, `y.npy`, `le.pkl`, `class_weights.npy`.

5. **Model Training**
   - `train_model.py`  
     Trains a PyTorch neural network on the embeddings.  
     Output: `emotion_classifier.pt`.

6. **Real-Time Prediction**
   - `app.py`  
     Streamlit app for predicting emotions from user input text.

---

## **Features**

* Preprocesses raw Reddit posts: cleans text, extracts emojis, and metadata (posting hour, text length).
* Maps 28 GoEmotions labels to **4 broad classes**: `positive`, `negative`, `mixed`, `neutral`.
* Converts text + emojis + metadata into numerical embeddings using **SentenceTransformer**.
* Trains a **PyTorch neural network** for emotion classification.
* Streamlit web app for **real-time emotion prediction** from user input.

---

## **Project Structure**

data/
├─ raw/
│ ├─ goemotions_1.csv
│ ├─ goemotions_2.csv
│ └─ goemotions_3.csv
└─ processed/
├─ social_posts.csv
├─ social_posts_processed.csv
├─ social_posts_sampled.csv
├─ social_posts_4class.csv
├─ X.npy
├─ y.npy
├─ le.pkl
└─ class_weights.npy

models/
└─ emotion_classifier.pt

src/
├─ preprocess.py # Data cleaning & preprocessing
├─ extract_features.py # Feature extraction (text, emoji, metadata)
├─ train_model.py # Train emotion classification model
└─ app.py # Streamlit app for predictions


## **Technologies Used**

* Python, Pandas, NumPy
* PyTorch
* Sentence Transformers (`all-MiniLM-L12-v2`)
* Streamlit
* NLTK for text preprocessing

---

## **Output**

* **Processed CSVs**: clean, preprocessed social media posts.
* **Numpy arrays**: `X.npy`, `y.npy` for features and labels.
* **Trained model**: `emotion_classifier.pt`.
* **Streamlit app**: real-time emotion prediction.

