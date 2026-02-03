## Deep Learning 

This repository contains **implementation-focused code and explanations strictly related to core Deep Learning architectures**:

* **Artificial Neural Networks (ANN)**
* **Convolutional Neural Networks (CNN)**
* **Recurrent Neural Networks (RNN)**
* **Long Short-Term Memory Networks (LSTM)**
* **Gated Recurrent Unit (GRU)**
* **DEEP RNN (recurrent neural network)**

---

## Table of Contents

1. Overview
2. Prerequisites
3. Artificial Neural Networks (ANN)
4. Convolutional Neural Networks (CNN)
5. Recurrent Neural Networks (RNN)
6. Long Short-Term Memory (LSTM)
7. Gated Recurrent unit (GRU)
8. DEEP RNN (Recurrent Neural Network)
9. Common Training Components
10. Evaluation Metrics
11. Repository Structure
12. How to Run the Code
13. Use Cases
14. Author

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

## 7. Gated Recurrent Unit (GRU)

*  A GRU is an advanced RNN architecture that uses gates to control the flow of information and reduce the vanishing gradient problem. In a Deep GRU, instead of a single GRU layer, two or more GRU layers are stacked, making the network deep in space while still recurrent in time.


## 8. DEEP RNN (Recurrent Neural Network)

* In a basic RNN, there is only one recurrent layer, which processes sequential data by maintaining a hidden state that captures information from previous time steps. In contrast, a Deep RNN consists of two or more recurrent layers, where the output (hidden state) of one layer at a given time step is used as the input to the next layer at the same time step.

  
## 9. Common Training Components

These components are shared across ANN, CNN, RNN, and LSTM implementations:

* Activation functions (ReLU, Sigmoid, Tanh, Softmax)
* Optimizers (SGD, Adam)
* Batch size and epochs
* Learning rate
* Overfitting and underfitting
* Validation split

---

## 10. Evaluation Metrics

### Classification

* Accuracy
* Precision
* Recall
* F1-score

### Regression

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

---

## 11. Repository Structure

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

## 12. How to Run the Code

1. Clone the repository
2. Install required dependencies
3. Navigate to the specific model folder (ANN/CNN/RNN/LSTM)
4. Run the Python file

---

## 13. Use Cases

* Academic practical files
* Deep Learning lab experiments
* Model comparison studies
* Interview demonstrations

---

