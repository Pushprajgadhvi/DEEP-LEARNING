# Deep Learning

This repository contains **implementation-focused notebooks and explanations covering core Deep Learning and Natural Language Processing architectures**, from basic neural networks to modern NLP pipelines.

It includes practical implementations of:

* Artificial Neural Networks (ANN)
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* LSTM and GRU
* Deep and Bidirectional RNN
* Attention concepts
* NLP preprocessing techniques
* Word embeddings (Word2Vec, TF-IDF, Bag-of-Words)
* Transfer Learning
* Named Entity Recognition (NER)
* Sentiment Analysis
* Data preprocessing and augmentation

---

# Table of Contents

1. Overview  
2. Prerequisites  
3. ANN  
4. CNN  
5. RNN  
6. LSTM  
7. GRU  
8. Deep RNN  
9. Bidirectional RNN  
10. NLP Preprocessing  
11. Feature Extraction (BOW, TF-IDF, n-grams)  
12. Word Embeddings  
13. Named Entity Recognition  
14. Sentiment Analysis  
15. Transfer Learning  
16. Training Components  
17. Evaluation Metrics  
18. Repository Structure  
19. How to Run  
20. Use Cases  
21. Author  

---

# 1. Overview

This repository provides **hands-on implementation notebooks** for understanding Deep Learning and NLP from fundamentals to advanced topics.

It focuses on:

* Concept clarity
* Implementation practice
* Academic and interview preparation
* Real dataset experimentation

---

# 2. Prerequisites

## Mathematics

* Linear Algebra
* Probability
* Gradients and derivatives

## Programming

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras / PyTorch
* Jupyter Notebook

---

# 3. Artificial Neural Networks (ANN)

Notebook:

```
Ann.ipynb
```

Concepts:

* Perceptron
* Dense layers
* Backpropagation
* Binary classification

Dataset:

```
Churn_Modelling.csv
```

Use case:

Customer churn prediction

---

# 4. Convolutional Neural Networks (CNN)

Notebooks:

```
CNNimp.ipynb
lenet.ipynb
pooling.ipynb
visualizing.ipynb
```

Concepts:

* Convolution layers
* Pooling
* Feature maps
* CNN architecture

---

# 5. Recurrent Neural Networks (RNN)

Notebooks:

```
Rnn_architecture.ipynb
DeepRnn.ipynb
```

Concepts:

* Sequential learning
* Hidden state
* Time-series modeling

---

# 6. Long Short Term Memory (LSTM)

Notebook:

```
LSTM.ipynb
```

Concepts:

* Forget gate
* Input gate
* Output gate
* Long-term dependency learning

---

# 7. GRU (Gated Recurrent Unit)

Notebook:

```
gru.ipynb
```

Concepts:

* Update gate
* Reset gate
* Efficient sequence learning

---

# 8. Bidirectional RNN

Notebook:

```
bidirectionRNN.ipynb
```

Concept:

Processes sequence forward and backward.

---

# 9. NLP Preprocessing

Notebooks:

```
STOPwords.ipynb
stemming.ipynb
padding.ipynb
integer_encoding_learn.ipynb
redex.ipynb
```

Concepts:

* Stopword removal
* Stemming
* Tokenization
* Padding sequences
* Text cleaning

---

# 10. Feature Extraction

## Bag of Words

```
BOW.ipynb
```

## Bag of n-grams

```
BOn-g.ipynb
```

## TF-IDF

```
TF-IDF.ipynb
```

Purpose:

Convert text into numerical vectors.

---

# 11. Word Embeddings

Notebooks:

```
wordv.ipynb
wordv2.ipynb
```

Concepts:

* Word2Vec
* Semantic similarity
* Dense vector representation

---

# 12. Named Entity Recognition (NER)

Notebook:

```
NER.ipynb
```

Concept:

Detect entities like:

Person  
Location  
Organization  

Using spaCy.

---

# 13. Sentiment Analysis

Notebooks:

```
sentimental_analysis.ipynb
simple_learn_sentimental.ipynb
```

Datasets:

```
spam.csv
news_dataset.json
Fake_Real_Data.csv
```

Use cases:

Spam detection  
Fake news detection  
Sentiment classification  

---

# 14. spaCy NLP

Notebooks:

```
spacy1.ipynb
spacy2.ipynb
```

Concepts:

Tokenization  
NER  
POS tagging  

---

# 15. Transfer Learning

Notebooks:

```
transferlearning.ipynb
transferlearning.finetuining.ipynb
transferlearningdataaug.ipynb
pretrainedmodels.ipynb
```

Concepts:

* Pretrained CNN models
* Fine tuning
* Feature extraction

---

# 16. Functional API Models

Notebooks:

```
functional_api_demo.ipynb
functional_multiple_input.ipynb
```

Concepts:

Multi-input models  
Complex architectures  

---

# 17. Data Augmentation

Notebook:

```
data_augmentation.ipynb
```

Purpose:

Increase dataset size artificially.

---

# 18. Datasets Included

```
Churn_Modelling.csv
Ecommerce_data.csv
Fake_Real_Data.csv
news_dataset.json
spam.csv
```

Used for:

Classification  
Sentiment analysis  
Fake news detection  

---

# 19. Repository Structure

```
DeepLearning/

│
├── ANN
│   ├── Ann.ipynb
│   ├── Churn_Modelling.csv
│
├── CNN
│   ├── CNNimp.ipynb
│   ├── lenet.ipynb
│   ├── pooling.ipynb
│   ├── visualizing.ipynb
│
├── RNN
│   ├── Rnn_architecture.ipynb
│   ├── DeepRnn.ipynb
│   ├── bidirectionRNN.ipynb
│   ├── gru.ipynb
│   ├── LSTM.ipynb
│
├── NLP
│   ├── BOW.ipynb
│   ├── BOn-g.ipynb
│   ├── TF-IDF.ipynb
│   ├── STOPwords.ipynb
│   ├── stemming.ipynb
│   ├── padding.ipynb
│   ├── integer_encoding_learn.ipynb
│   ├── redex.ipynb
│   ├── NER.ipynb
│   ├── spacy1.ipynb
│   ├── spacy2.ipynb
│   ├── sentimental_analysis.ipynb
│   ├── simple_learn_sentimental.ipynb
│   ├── wordv.ipynb
│   ├── wordv2.ipynb
│
├── TransferLearning
│   ├── transferlearning.ipynb
│   ├── transferlearning.finetuining.ipynb
│   ├── transferlearningdataaug.ipynb
│   ├── pretrainedmodels.ipynb
│
├── FunctionalAPI
│   ├── functional_api_demo.ipynb
│   ├── functional_multiple_input.ipynb
│
├── datasets
│   ├── spam.csv
│   ├── news_dataset.json
│   ├── Fake_Real_Data.csv
│   ├── Ecommerce_data.csv
│
└── README.md
```

---

# 20. How to Run

Install requirements:

```
pip install numpy pandas matplotlib tensorflow scikit-learn spacy
```

Run notebook:

```
jupyter notebook
```

Open desired `.ipynb` file.

---

# 21. Use Cases

Deep learning practice  
NLP learning  
University labs  
Interview preparation  
Portfolio projects  
