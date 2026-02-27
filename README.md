# Deep Learning

This repository contains **implementation-focused code and explanations strictly related to core Deep Learning architectures**:

* **Artificial Neural Networks (ANN)**
* **Convolutional Neural Networks (CNN)**
* **Recurrent Neural Networks (RNN)**
* **Long Short-Term Memory Networks (LSTM)**
* **Gated Recurrent Unit (GRU)**
* **DEEP RNN (recurrent neural network)**
* **Bidirectional RNN**
* **Attention Mechanism**
* **Bahdanau Attention**
* **Luong Attention**
* **Transformer Architecture**
* **Self-Attention**
* **Multi-Head Attention**
* **Positional Encoding**
* **Encoder–Decoder Architecture**
* **Masked Attention**
* **BERT (Bidirectional Encoder Representations from Transformers)**
* **GPT (Generative Pre-trained Transformer)**
* **Large Language Models (LLMs)**
* **Natural Language Processing (NLP)**

---

# Table of Contents

1. Overview  
2. Prerequisites  
3. Artificial Neural Networks (ANN)  
4. Convolutional Neural Networks (CNN)  
5. Recurrent Neural Networks (RNN)  
6. Long Short-Term Memory (LSTM)  
7. Gated Recurrent Unit (GRU)  
8. DEEP RNN  
9. Bidirectional RNN  
10. Attention Mechanism  
11. Bahdanau Attention  
12. Luong Attention  
13. Transformer Architecture  
14. Self-Attention Mechanism  
15. Multi-Head Attention  
16. Positional Encoding  
17. Encoder–Decoder Architecture  
18. Masked Attention  
19. BERT  
20. GPT  
21. Large Language Models (LLMs)  
22. Common Training Components  
23. Evaluation Metrics  
24. Natural Language Processing (NLP)  
25. Repository Structure  
26. How to Run the Code  
27. Use Cases  
28. Author  

---

# 1. Overview

This repository is designed for **students and practitioners** who want a **clear, practical understanding of Deep Learning architectures**, from basic neural networks to modern Transformer-based models.

It covers:

* Classical Deep Learning (ANN, CNN, RNN, LSTM, GRU)
* Sequence modeling and Attention
* Transformer architecture
* Modern NLP models (BERT, GPT)
* Foundations of Large Language Models

Suitable for:

* University coursework and lab submissions  
* Concept clarification  
* Interview preparation  
* Deep Learning and NLP projects  

---

# 2. Prerequisites

## Mathematical Basics

* Linear Algebra (vectors, matrices)
* Probability and Statistics
* Derivatives and Gradients
* Matrix multiplication intuition

## Programming & Tools

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras or PyTorch

---

# 3. Artificial Neural Networks (ANN)

## Concepts Covered

* Biological inspiration
* Perceptron
* Dense layers
* Forward propagation
* Backpropagation
* Loss functions
* Gradient descent

## Use Cases

* Tabular classification
* Regression
* Pattern recognition

---

# 4. Convolutional Neural Networks (CNN)

## Concepts Covered

* Convolution operation
* Filters and feature maps
* Padding and stride
* Pooling layers
* Flatten and dense layers

## Use Cases

* Image classification
* Object detection
* Computer vision tasks

---

# 5. Recurrent Neural Networks (RNN)

## Concepts Covered

* Sequential processing
* Hidden state memory
* Time-step learning
* Many-to-many models
* Vanishing gradient problem

## Use Cases

* Time series prediction
* Text classification

---

# 6. Long Short-Term Memory (LSTM)

## Concepts Covered

* Long-term memory handling
* Cell state
* Gates:
  * Forget gate
  * Input gate
  * Output gate

## Use Cases

* NLP
* Forecasting
* Speech recognition

---

# 7. Gated Recurrent Unit (GRU)

GRU simplifies LSTM by using:

* Update gate
* Reset gate

Advantages:

* Faster training
* Less parameters
* Good performance

---

# 8. DEEP RNN

Stacking multiple RNN layers increases learning capacity.

Benefits:

* Better abstraction
* Improved sequence modeling

---

# 9. Bidirectional RNN

Processes sequence:

Forward →  
Backward ←  

Captures:

* Past context
* Future context

---

# 10. Attention Mechanism

Attention allows models to focus on important words instead of compressing all information into one vector.

Benefits:

* Better long sequence handling
* Improved translation
* Improved NLP performance

---

# 11. Bahdanau Attention

Also called:

Additive Attention

Key idea:

Alignment score calculated using neural network.

Advantages:

Better performance for long sequences.

---

# 12. Luong Attention

Also called:

Multiplicative Attention

Advantages:

* Faster computation
* Efficient implementation

---

# 13. Transformer Architecture

Transformer is a deep learning architecture introduced in 2017.

Core idea:

**Uses attention instead of recurrence or convolution**

Main components:

Encoder  
Decoder  
Attention layers  
Feedforward network  

Advantages:

Parallel processing  
Faster training  
Better performance  

Used in:

BERT  
GPT  
Modern LLMs  

---

# 14. Self-Attention Mechanism

Self-attention allows each word to look at other words in the same sentence.

Key components:

Query (Q)  
Key (K)  
Value (V)

Formula:

Attention(Q,K,V) = softmax(QKᵀ / √d) V

Benefits:

Captures relationships between words.

---

# 15. Multi-Head Attention

Instead of one attention, multiple attentions run in parallel.

Benefits:

Model learns:

Syntax relationships  
Semantic relationships  
Different context types  

---

# 16. Positional Encoding

Transformers have no sequence order awareness.

Positional encoding adds position information.

Uses:

Sine and cosine functions

Purpose:

Helps model understand word order.

---

# 17. Encoder–Decoder Architecture

Encoder:

Processes input sequence.

Decoder:

Generates output sequence.

Used in:

Translation  
Chatbots  
Text generation  

---

# 18. Masked Attention

Prevents model from seeing future tokens.

Used in:

GPT  
Text generation  

Ensures:

Autoregressive prediction

---

# 19. BERT

Full form:

Bidirectional Encoder Representations from Transformers

Architecture:

Encoder-only Transformer

Key features:

Bidirectional context  
Masked language modeling  

Uses:

Search engines  
Text classification  
Question answering  

---

# 20. GPT

Full form:

Generative Pre-trained Transformer

Architecture:

Decoder-only Transformer

Key features:

Text generation  
Autoregressive prediction  

Uses:

Chatbots  
Content generation  
Coding assistants  

---

# 21. Large Language Models (LLMs)

LLMs are very large Transformer models trained on massive text datasets.

Examples:

GPT  
BERT  
LLaMA  

Capabilities:

Text generation  
Reasoning  
Translation  
Summarization  

Training stages:

Pretraining  
Fine-tuning  
Inference  

---

# 22. Common Training Components

Shared across models:

Activation functions:

ReLU  
Sigmoid  
Tanh  
Softmax  

Optimizers:

SGD  
Adam  

Hyperparameters:

Learning rate  
Batch size  
Epochs  

Problems:

Overfitting  
Underfitting  

Solutions:

Regularization  
Dropout  

---

# 23. Evaluation Metrics

## Classification

Accuracy  
Precision  
Recall  
F1 score  

## Regression

MAE  
MSE  

## NLP Metrics

BLEU score  
Perplexity  

---

# 24. Natural Language Processing (NLP)

NLP enables machines to understand human language.

Tasks:

Text classification  
Translation  
Chatbots  
Summarization  

Modern NLP uses:

RNN  
LSTM  
Transformers  
LLMs  

---

# 25. Repository Structure

```
ANN/
│── ann_model.py
│── dataset.csv
│── README.md

CNN/
│── cnn_model.py
│── image_dataset/
│── README.md

RNN/
│── rnn_model.py
│── sequence_data.csv
│── README.md

LSTM/
│── lstm_model.py
│── time_series_data.csv
│── README.md

GRU/
│── gru_model.py

TRANSFORMER/
│── transformer_model.py

BERT/
│── bert_explanation.md

GPT/
│── gpt_explanation.md
```

---

# 26. How to Run the Code

Step 1:

Clone repository

```
git clone https://github.com/yourusername/deep-learning
```

Step 2:

Install dependencies

```
pip install numpy pandas matplotlib tensorflow torch
```

Step 3:

Run model

```
python ann_model.py
```

---

# 27. Use Cases

Academic practical files  
Deep learning experiments  
NLP learning  
Transformer understanding  
Interview preparation  

---

# 28. Author

Deep Learning Repository  
Created for learning and implementation of modern Deep Learning architectures.
