# App Store Sentiment Analysis — TaHoma by Somfy

Sentiment analysis project on user reviews of the TaHoma by Somfy application, collected from the Apple App Store. The goal is to automatically classify reviews as Positive, Neutral, or Negative using a set of NLP approaches ranging from classical Machine Learning to fine-tuned Transformer models.

---

## Table of Contents

- [Context](#context)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Demo App](#demo-app)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Tech Stack](#tech-stack)

---

## Context

This project was carried out as part of a final-year study on NLP applied to customer feedback in French. The TaHoma app is a smart home controller by Somfy, and the reviews collected reflect a wide range of user experiences — from satisfaction to technical frustrations — which makes the classification task both realistic and challenging.

Ten models were implemented and compared across three families: Machine Learning, Deep Learning, and Transformers. A Streamlit web app was also built to allow real-time prediction using the best deep learning model.

---

## Dataset

- **Source**: Apple App Store (TaHoma by Somfy)
- **Size**: 500 reviews
- **Language**: French
- **Period**: September 2025 – April 2026

### Columns

| Column | Description |
|---|---|
| Date | Publication date |
| Auteur | Reviewer username |
| Note | Star rating (1 to 5) |
| Version | App version at review time |
| Titre | Review title |
| Commentaire | Full review text |
| Votes | Helpfulness votes |

### Labeling rule

Star ratings were converted to sentiment labels as follows:

| Rating | Sentiment | Count | Share |
|---|---|---|---|
| 1 – 2 | Negative | 185 | 37.0% |
| 3 | Neutral | 104 | 20.8% |
| 4 – 5 | Positive | 211 | 42.2% |

---

## Preprocessing

The full pipeline is documented in `Data_Preprocessing_&_NLP_Pipeline.ipynb`. Steps applied to raw French text:

1. Lowercase conversion
2. Emoji removal
3. URL removal
4. Special character removal
5. Accent normalization (Unidecode)
6. Stopword removal with SpaCy (French model)
7. Whitespace cleanup

For Machine Learning models, the cleaned text was then vectorized with TF-IDF. For Deep Learning models, Keras tokenization and sequence padding were used instead.

---

## Models

### Machine Learning

Each model was tested in three configurations: standard, class-weighted, and with SMOTE oversampling to handle the class imbalance.

- Naive Bayes
- Logistic Regression
- Support Vector Machine (LinearSVC)
- Random Forest
- XGBoost

### Deep Learning

- LSTM
- BiLSTM
- CNN + LSTM (hybrid architecture)

### Transformers

- DistilBERT (multilingual, fine-tuned)
- CamemBERT (French BERT, fine-tuned)

---

## Results

All models were evaluated on a held-out 20% test set (100 reviews).

### Machine Learning

| Model | Best Accuracy | Configuration |
|---|---|---|
| Naive Bayes | 58% | Baseline |
| Logistic Regression | 60% | Baseline |
| SVM | 55% | Class Weighted |
| Random Forest | 63% | SMOTE |
| XGBoost | 61% | Class Weighted |

### Deep Learning

| Model | Test Accuracy |
|---|---|
| LSTM | 61% |
| BiLSTM | 60% |
| CNN + LSTM | 59% |

### Transformers

| Model | Test Accuracy |
|---|---|
| DistilBERT | 52% |
| CamemBERT | 59% |

### Notes

Random Forest with SMOTE achieved the best overall accuracy at 63%. Deep learning models performed comparably despite the relatively small dataset size. Transformer models underperformed here — which is expected when fine-tuning on only 500 samples, as these architectures require significantly more data to generalize well.

---

## Demo App

A Streamlit application was developed to run predictions in real time. It loads the pre-trained BiLSTM model and allows the user to type any review text and instantly see the predicted sentiment along with the confidence score and full probability breakdown per class.

To launch it:

```bash
streamlit run app.py
```

The model files required are:

- `bilstm_sentiment_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

---

## Project Structure

```
App-Store-Sentiment-Analysis/
|
|-- Data_Preprocessing_&_NLP_Pipeline.ipynb
|
|-- Machine Learning/
|   |-- ML_NaiveBayes.ipynb
|   |-- ML_Regression_Logistique.ipynb
|   |-- ML_SVM.ipynb
|   |-- ML_RandomForest.ipynb
|   `-- ML_XGboost.ipynb
|
|-- Deep Learning/
|   |-- LSTM_BiLSTM_CNN.ipynb
|   |-- DistilBERT.ipynb
|   `-- CamemBERT.ipynb
|
|-- app.py
|-- bilstm_sentiment_model.h5
|-- tokenizer.pkl
|-- label_encoder.pkl
|-- requirements.txt
`-- README.md
```

---

## Installation

```bash
git clone https://github.com/filaliamine8/App-Store-Sentiment-Analysis.git
cd App-Store-Sentiment-Analysis

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

The notebooks were developed on Google Colab. To run them, upload the dataset file and execute cells in order.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | Pandas, NumPy |
| NLP | SpaCy, Regex, Emoji, Unidecode |
| Machine Learning | Scikit-learn, XGBoost, imbalanced-learn |
| Deep Learning | TensorFlow / Keras |
| Transformers | Hugging Face Transformers |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
