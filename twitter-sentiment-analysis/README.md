# 🐦 Twitter Sentiment Analysis using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-6A5ACD)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-9C27B0)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

An end-to-end **Natural Language Processing (NLP)** project that performs **sentiment analysis on Twitter data**
using classical machine learning techniques and a complete text preprocessing pipeline.

---

## 📌 Overview

This project implements a **Twitter Sentiment Analysis system** that classifies tweets into **positive** or
**negative** sentiment using **TF-IDF feature extraction** and a **Multinomial Naïve Bayes classifier**.

The focus of this work is on **text preprocessing, feature engineering, and model evaluation**, rather than
deployment or API integration.

---

## 🎯 Objectives

- Perform sentiment analysis on real-world Twitter data  
- Apply standard NLP preprocessing techniques  
- Convert text into numerical features using **TF-IDF**  
- Train and evaluate a **Multinomial Naïve Bayes** model  

---

## 📂 Dataset

- **Dataset Name:** Sentiment140  
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/kazanova/sentiment140 
- **Size:** 1.6 million tweets  

### Label Mapping
- `0` → Negative  
- `4` → Positive  

Neutral tweets were excluded.  
The dataset was **balanced to 200,000 tweets** (100k positive + 100k negative).

---

## 🛠 NLP Pipeline

The following preprocessing steps were applied:

- Lowercasing text  
- Removing URLs, mentions, hashtags, and punctuation  
- Tokenization  
- Stopword removal  
- Stemming (Porter Stemmer)  
- Lemmatization (WordNet Lemmatizer)  

---

## 📊 Feature Engineering

- **TF-IDF Vectorization**
  - Maximum features: `10,000`
  - N-grams: `(1, 2)` (unigrams + bigrams)

---

## 🤖 Model

- **Multinomial Naïve Bayes**
- Trained on TF-IDF features
- Stratified train-test split (80% train / 20% test)

---

## 📈 Evaluation

Evaluation was performed using:

- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  

**Test Accuracy:** approximately **74%**

---

## 🎓 Academic Context

This project was completed as part of a **Natural Language Processing (NLP) course**, demonstrating:

- Practical NLP pipelines  
- Machine learning-based sentiment classification  
- Feature extraction and evaluation techniques  

---

## 👤 Author

**Sajidur Rahman Sajid**  
Computer Science & Engineering (CSE)  
Aspiring **AI / Machine Learning / NLP Engineer**
