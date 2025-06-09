# Day 8 

## Overview

In this session, we focused on building a neural network model using **PyTorch** and explored key concepts such as **activation functions**, **optimizers**, and **loss functions**. We learned how to create a random dataset, build a custom neural network model from scratch, and then transitioned to using PyTorch's built-in functionalities for optimization and training.

---

## Table of Contents

1. [Create a Random Dataset](#create-a-random-dataset)
2. [Implement a Custom Neural Network](#implement-a-custom-neural-network)
3. [Using Built-in Functions for Loss and Optimizer](#using-built-in-functions-for-loss-and-optimizer)
4. [Understand Activation Functions](#understand-activation-functions)

---

## 1. **Create a Random Dataset**

We started by creating a synthetic dataset for training. The dataset simulated environmental conditions such as **temperature**, **rainfall**, and **humidity** as input features, and the corresponding **targets** were the amounts of **apples** and **oranges** produced.

### Dataset:
- **Inputs**: Random data representing weather conditions.
- **Targets**: Corresponding values for apples and oranges.

We used the following **random values** for the dataset:
- **Inputs**: 3 columns (temperature, rainfall, humidity) with 10 data points.
- **Targets**: 2 columns (apples, oranges) with 10 corresponding output values.

This dataset was used to train a simple neural network model to predict apple and orange quantities from environmental data.

---

## 2. **Implement a Custom Neural Network**

In the first approach, we defined a simple **fully connected neural network** without using high-level PyTorch modules like `nn.Module`. The model was built manually with:
- A **linear transformation** to calculate the weighted sum of inputs (`x @ w + b`).
- A **custom loss function** (Mean Squared Error or MSE) to measure how well the predictions matched the targets.
- **Gradient Descent** was implemented manually to adjust the weights and biases during training.

The model was trained using **backpropagation** to update the weights:
- **Gradients** were computed using `loss.backward()`.
- The weights were manually updated by subtracting the gradients scaled by a small learning rate.

---

## 3. **Using Built-in Functions for Loss and Optimizer**

After manually implementing the training loop, we transitioned to using **PyTorch's built-in functionalities** to simplify the process:
- **Model**: A more structured approach was used by building a custom neural network class (`SimpleNN`) using `nn.Module`.
- **Loss Function**: We replaced our custom MSE loss with **Mean Absolute Error (MAE)** using PyTorch’s built-in `torch.nn.L1Loss()`.
- **Optimizer**: Instead of manually updating the weights, we used the **Adam optimizer** (`torch.optim.Adam`) to optimize the weights during training.

We also explored the use of **Stochastic Gradient Descent (SGD)** as an alternative optimizer. This helped us see how different optimizers influence the training process and model performance.

---

## 4. **Understand Activation Functions**

A crucial part of neural networks is the use of **activation functions**, which introduce **non-linearity** to the model, allowing it to learn more complex patterns. We covered the following activation functions:

### 1. **Sigmoid**
- **Use Case**: Typically used in binary classification problems, especially in the output layer.
- **Pros**: Smooth gradient, probabilistic outputs.
- **Cons**: Prone to vanishing gradients for large input values.

### 2. **Tanh**
- **Use Case**: Often used in hidden layers; outputs values between -1 and 1.
- **Pros**: Zero-centered output.
- **Cons**: Like Sigmoid, suffers from vanishing gradients for extreme values.

### 3. **ReLU (Rectified Linear Unit)**
- **Use Case**: Commonly used in hidden layers due to its simplicity and effectiveness.
- **Pros**: Avoids vanishing gradients, computationally efficient.
- **Cons**: Can lead to the **dying ReLU problem** where neurons get stuck and stop updating.

### 4. **Leaky ReLU**
- **Use Case**: Used to address the dying ReLU problem by allowing a small negative slope for values less than 0.
- **Pros**: Prevents dying neurons.
- **Cons**: Still prone to some issues with large values.

### 5. **Softmax**
- **Use Case**: Typically used in the output layer for multi-class classification problems.
- **Pros**: Converts raw output values into a probability distribution.
- **Cons**: Sensitive to outliers in logits, can cause instability with extreme values.

---

## Key Concepts Learned Today

1. **Activation Functions**:
   - We explored how each activation function affects the learning and performance of the model.
   - Learned when and why to use specific functions for different layers (e.g., **ReLU** for hidden layers, **Softmax** for multi-class output).

2. **Optimizers**:
   - We learned how optimizers like **Adam** and **SGD** help update the weights during training by using **gradient information** and adjusting the learning rate.
   - Understanding the trade-offs between different optimizers and their impact on training speed and model performance.

3. **Training with Backpropagation**:
   - We understood how **backpropagation** works by computing gradients and updating model weights accordingly.
   - Switched from a manual gradient update to using **PyTorch optimizers**, simplifying the process.

---

