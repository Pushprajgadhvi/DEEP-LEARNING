# Deep Learning 

This repository contains **implementation-focused code and explanations strictly related to core Deep Learning architectures**:

* **Artificial Neural Networks (ANN)**
* **Convolutional Neural Networks (CNN)**
* **Recurrent Neural Networks (RNN)**
* **Long Short-Term Memory Networks (LSTM)**
*  **Gated Recurrent Unit (GRU)**

---

## Table of Contents

1. Overview
2. Prerequisites
3. Artificial Neural Networks (ANN)
4. Convolutional Neural Networks (CNN)
5. Recurrent Neural Networks (RNN)
6. Long Short-Term Memory (LSTM)
7. Common Training Components
8. Evaluation Metrics
9. Repository Structure
10. How to Run the Code
11. Use Cases
12. Author

---

## 1. Overview

This repository is designed for **students and practitioners** who want a **clear, practical understanding of ANN, CNN, RNN, and LSTM**, supported by working code implementations. It is suitable for:

* University coursework and lab submissions
* Concept clarification
* Interview preparation
* Entry-level Deep Learning projects

---

## 2. Prerequisites

### Mathematical Basics

* Linear Algebra (vectors, matrices)
* Basic Probability and Statistics
* Derivatives and gradients

### Programming & Tools

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras or PyTorch (as used in the code)

---

## 3. Artificial Neural Networks (ANN)

### Concepts Covered

* Biological inspiration of neural networks
* Artificial neuron (Perceptron)
* Fully Connected (Dense) layers
* Forward propagation
* Backpropagation
* Loss functions

  * Mean Squared Error (MSE)
  * Binary & Categorical Cross-Entropy
* Gradient Descent

### Typical Use Cases

* Tabular data classification
* Regression problems
* Basic pattern recognition

---

## 4. Convolutional Neural Networks (CNN)

### Concepts Covered

* Convolution operation
* Kernels / filters
* Feature maps
* Padding and stride
* Pooling layers (Max & Average)
* Flattening and Dense layers
* CNN training pipeline

### Common Architectures Implemented

* Basic CNN
* Deep CNN with multiple convolution blocks

### Typical Use Cases

* Image classification
* Handwritten digit recognition
* Object and pattern detection

---

## 5. Recurrent Neural Networks (RNN)

### Concepts Covered

* Sequential data modeling
* Time-step based processing
* Hidden state representation
* Many-to-one and many-to-many models
* Vanishing gradient problem

### Typical Use Cases

* Time series prediction
* Text classification
* Sequence modeling tasks

---

## 6. Long Short-Term Memory (LSTM)

### Concepts Covered

* Motivation for LSTM over RNN
* Cell state and hidden state
* Gates:

  * Forget gate
  * Input gate
  * Output gate
* Handling long-term dependencies

### Typical Use Cases

* Sentiment analysis
* Language modeling
* Stock price and time-series forecasting

---

## 7. Common Training Components

These components are shared across ANN, CNN, RNN, and LSTM implementations:

* Activation functions (ReLU, Sigmoid, Tanh, Softmax)
* Optimizers (SGD, Adam)
* Batch size and epochs
* Learning rate
* Overfitting and underfitting
* Validation split

---

## 8. Evaluation Metrics

### Classification

* Accuracy
* Precision
* Recall
* F1-score

### Regression

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

---

## 9. Repository Structure

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
```

---

## 10. How to Run the Code

1. Clone the repository
2. Install required dependencies
3. Navigate to the specific model folder (ANN/CNN/RNN/LSTM)
4. Run the Python file

---

## 11. Use Cases

* Academic practical files
* Deep Learning lab experiments
* Model comparison studies
* Interview demonstrations

---

