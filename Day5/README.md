# Day 5 - AI & ML Foundations with AWS

Today, I decided to enroll in foundational AI/ML courses to deepen my understanding and strengthen my skills further. This will help me solidify my knowledge before diving into more advanced topics.

File Directory: 
```bash
├── Day5/
│   ├── AWS_AI_ML/
│   │   ├── 4_1_Supervised_Learning.ipynb
│   │   ├── 4_2_Unsupervised_Learning.ipynb
│   │   ├── 4_3_Decision_Tree.ipynb
│   │   └── 4_4_Neural_Network.ipynb
│   └── README.md
```


---

## Supervised Learning

### 🧠 **Project Summary: Building Energy Efficiency Prediction**

- **Dataset**: Generated a synthetic dataset of 500 samples with the following features:
  - `WallArea`, `RoofArea`, `OverallHeight`, `GlazingArea`
  - **Target**: `EnergyEfficiency`
- **Data Distribution**: All features use uniform distributions.
- **Visualization**: Visualized feature distributions and relationships using `seaborn`.
- **Model**: Trained a **Random Forest Regressor** to predict energy efficiency.
- **Evaluation**: Evaluated the model using **Mean Squared Error** (MSE) and visualized predictions vs true values in a scatter plot.

---

## Unsupervised Learning

### 🚗 **Project Summary: Vehicle Clustering with K-Means**

- **Dataset**: Created a synthetic dataset of 300 vehicles with the following features:
  - `Weight`, `EngineSize`, and `Horsepower`
- **Clustering**: Applied **KMeans clustering** with 3 clusters.
- **Visualization**: Visualized the clusters using a scatter plot of **Weight vs Horsepower**.

---

## Decision Tree Example

### 🚗 **Project Summary: Churn Prediction**

- **Node Details**:
  - **Gini Impurity**: 0.48
  - **Samples**: 120
  - **Values**: [60, 60] (split evenly between "No Churn" and "Churn")
  - **Class**: "No Churn" (majority class)
  - **Feature Used for Split**: "Age > 40"

**Interpretation**:
- This node has 120 samples split evenly between "No Churn" and "Churn", resulting in a **Gini impurity** of 0.48.
- The data is split based on the feature "Age > 40". Since "No Churn" is the majority class, the predicted class for this node is "No Churn".

---

## Neural Network for Binary Classification: Purchase Prediction

### 🚀 **Project Summary: Predicting Purchases Based on User Behavior**

- **Objective**: Predict whether a user will make a purchase based on their `VisitDuration` and `PagesVisited`.
- **Steps**:
  1. **Data Generation**: Created synthetic data with 200 samples, labeled as `Purchase` (1) or `No Purchase` (0).
  2. **Preprocessing**: Split the data into training (80%) and testing (20%) sets.
  3. **Model**: Built a simple neural network with:
     - 1 hidden layer (10 neurons, ReLU activation)
     - Output layer with 1 neuron (sigmoid activation for binary classification)
  4. **Training**: Trained the model for 10 epochs.
  5. **Evaluation**: Evaluated the model on the test set and printed the accuracy.

**Outcome**: Successfully trained a neural network to predict purchases based on user behavior with binary classification.

---
