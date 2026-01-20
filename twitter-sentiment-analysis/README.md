# 🐦 Twitter Sentiment Analysis using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Processing-4CAF50)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-9C27B0)
![Academic Project](https://img.shields.io/badge/Academic_Project-Yes-2E7D32)

---

## 📌 Overview

This project implements a **Twitter Sentiment Analysis system** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The goal is to classify tweets into **positive** or **negative** sentiment by applying a complete NLP pipeline followed by a **Naïve Bayes classifier**.

The project was developed as part of an **academic NLP course** and focuses on **text preprocessing, feature extraction, and model evaluation**, rather than deployment.

---

## 🎯 Objectives

- Perform end-to-end sentiment analysis on real-world Twitter data  
- Apply standard NLP preprocessing techniques  
- Convert text into numerical features using **TF-IDF**
- Train and evaluate a **Multinomial Naïve Bayes** model
- Analyze model performance using accuracy, classification report, and confusion matrix

---

## 📂 Dataset

- **Dataset Name:** Sentiment140  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/kazanova/sentiment140

### Dataset Details
- 1.6 million tweets
- Sentiment labels:
  - `0` → Negative
  - `4` → Positive
- Neutral tweets are excluded
- Balanced to **200,000 tweets** (100k positive + 100k negative)

The processed and balanced dataset is saved as:
balanced_sentiment140_no_neutral.csv


---

## 🛠️ NLP Pipeline

The following preprocessing steps were applied:

1. Lowercasing text  
2. Removing URLs, mentions, hashtags, and special characters  
3. Removing punctuation  
4. Tokenization  
5. Stopword removal  
6. Synonym substitution using **WordNet**  
7. Stemming (Porter Stemmer)  
8. Lemmatization (WordNet Lemmatizer)  
9. Final text reconstruction  

---

## 📊 Feature Engineering

- **TF-IDF Vectorization**
  - Max features: `10,000`
  - N-grams: `(1, 2)` (unigrams + bigrams)

---

## 🤖 Model Used

- **Multinomial Naïve Bayes**
- Trained using TF-IDF features
- Stratified train-test split (80% train / 20% test)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix visualization

The trained model achieves an accuracy of approximately **74%** on the test set.

---

## 🧪 Example Prediction

```text
Input:  "I hate this"
Output: Negative

Input:  "good"
Output: Positive


---

## 🛠️ NLP Pipeline

The following preprocessing steps were applied:

1. Lowercasing text  
2. Removing URLs, mentions, hashtags, and special characters  
3. Removing punctuation  
4. Tokenization  
5. Stopword removal  
6. Synonym substitution using **WordNet**  
7. Stemming (Porter Stemmer)  
8. Lemmatization (WordNet Lemmatizer)  
9. Final text reconstruction  

---

## 📊 Feature Engineering

- **TF-IDF Vectorization**
  - Max features: `10,000`
  - N-grams: `(1, 2)` (unigrams + bigrams)

---

## 🤖 Model Used

- **Multinomial Naïve Bayes**
- Trained using TF-IDF features
- Stratified train-test split (80% train / 20% test)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix visualization

The trained model achieves an accuracy of approximately **74%** on the test set.

---

## 🧪 Example Prediction

```text
Input:  "I hate this"
Output: Negative

Input:  "good"
Output: Positive

---

## 📁 Project Structure


twitter-sentiment-analysis/
│
├── data/
│   ├── raw/
│   │   └── training.1600000.processed.noemoticon.csv
│   └── processed/
│       └── balanced_sentiment140_no_neutral.csv
│
├── report/
│   └── twitter-sentiment-analysis.pdf
│
├── twitter_sentiment_analysis.ipynb
└── README.md

---

## 🎓 Academic Context

This project was completed as part of an **NLP course**, demonstrating practical understanding of:

- Text preprocessing techniques  
- NLP feature extraction  
- Machine learning-based text classification  
- Model evaluation and analysis  

---

## ⚠️ Notes

- This project is intended for **educational and learning purposes**
- No deployment or API integration is included
- Dataset credits belong to the original creators (**Sentiment140**)

---

## 👤 Author

**Sajidur Rahman Sajid**  
Computer Science & Engineering (CSE)  
Interested in **AI / Machine Learning / NLP**

---

⭐ Feel free to explore the notebook and documentation to understand the full workflow.
