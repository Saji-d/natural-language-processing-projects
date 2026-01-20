# 📊 Financial Sentiment Analysis using BERT and FinBERT

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-4CAF50)
![Academic Project](https://img.shields.io/badge/Academic_Project-Yes-6A1B9A)

---

## 📌 Overview

This project presents a **comparative study of BERT and FinBERT** for **financial sentiment classification**, focusing on how **domain-specific pretraining** impacts model performance.

The task involves classifying financial text into **positive, neutral, and negative sentiment**, using transformer-based models fine-tuned under identical experimental settings for a fair comparison.

This work was completed as part of an **academic Natural Language Processing (NLP) course**.

---

## 🎯 Objectives

- Perform sentiment analysis on financial text data  
- Compare **general-purpose BERT** vs **domain-specific FinBERT**  
- Evaluate performance using **Accuracy** and **Macro-F1 score**  
- Analyze the effect of financial-domain pretraining  

---

## 🧠 Models Used

### 🔹 BERT (Baseline)
- Model: `bert-base-uncased`
- Pretrained on general-domain corpora
- Used as a benchmark model

### 🔹 FinBERT (Domain-Specific)
- Model: `ProsusAI/finbert`
- Pretrained on financial news and reports
- Better captures financial language and sentiment

---

## 📂 Datasets

The experiments were conducted using **publicly available financial sentiment datasets**:

### 🔹 Financial PhraseBank–like Dataset
- Financial news statements and short texts
- Labels: `positive`, `neutral`, `negative`

### 🔹 FiQA-style Financial Dataset
- Financial questions, reports, and market-related text
- Labels: `positive`, `neutral`, `negative`

All datasets were **cleaned, normalized, and label-mapped consistently** before training.

---

## ⚙️ Experimental Setup

- **Platform:** Google Colab (GPU enabled)
- **Language:** Python 3
- **Framework:** PyTorch
- **Libraries:** Hugging Face Transformers, Datasets, Scikit-learn, NumPy, Pandas, Matplotlib

### Training Configuration
- Epochs: 3  
- Learning Rate: 2e-5  
- Batch Size: 16  
- Max Sequence Length: 128  
- Optimizer: AdamW  
- Loss Function: Cross Entropy  

---

## 📈 Results Summary

- **FinBERT consistently outperformed BERT**
- Higher **Accuracy** and **Macro-F1 score**
- Reduced confusion between neutral and positive classes
- Stronger understanding of financial context

📌 **Key Insight:**  
Domain-specific pretraining significantly improves sentiment analysis performance in financial text.

---

## 📊 Evaluation Metrics

- Accuracy  
- Macro-F1 Score  
- Confusion Matrix Analysis  

---

## 📄 Documentation

A detailed project report including:
- Dataset description  
- Model architecture  
- Training methodology  
- Experimental results  
- Confusion matrix analysis  

📘 **Report:**  
`financial_sentiment_bert_vs_finbert.pdf`

---

## 🎓 Academic Context

This project was developed as part of an **NLP course** to demonstrate:
- Transformer-based sentiment analysis
- Fair experimental comparison
- Financial-domain NLP modeling
- Research-style evaluation and reporting

---

## 👤 Author

**Sajidur Rahman Sajid**  
Computer Science & Engineering (CSE)  
American International University–Bangladesh (AIUB)

---

## 📜 License

This repository is shared for **educational and academic purposes**.
