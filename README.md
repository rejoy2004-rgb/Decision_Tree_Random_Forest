# 🌳 Decision Trees & Random Forests (Heart Disease Prediction)

## 📌 Overview

This project implements **Decision Tree** and **Random Forest** models to predict heart disease using a dataset. It also explores overfitting, feature importance, and model evaluation techniques.

---

## 🎯 Objectives

* Train a Decision Tree classifier
* Control overfitting using tree depth
* Train a Random Forest model
* Compare model performance
* Analyze feature importance
* Evaluate using cross-validation

---

## 🛠️ Tools & Libraries

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 📊 Dataset

* Heart Disease Dataset (`heart.csv`)
* Contains medical attributes like age, cholesterol, chest pain type, etc.

---

## ⚙️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/rejoy2004-rgb/Decision-Tree-Random-Forest.git
cd Decision-Tree-Random-Forest
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Code

```bash
python main.py
```

---

## 📈 Models Used

### 🌳 Decision Tree

* Simple and interpretable model
* Can overfit if depth is not controlled

### 🌲 Random Forest

* Ensemble of multiple decision trees
* Reduces overfitting
* Provides better accuracy

---

## 🔍 Key Features

### ✔ Overfitting Control

* Used `max_depth` to limit tree size

### ✔ Feature Importance

* Identified most important health indicators

### ✔ Cross Validation

* Used 5-fold cross-validation for reliable performance

---

## 📊 Results (Example)

* Decision Tree Accuracy: ~80-85%
* Random Forest Accuracy: ~85-90%
* Random Forest performs better due to ensemble learning

---

## 📌 Learnings

* Tree-based models are powerful and easy to interpret
* Random Forest improves generalization
* Feature importance helps in understanding data

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Add visualization using Graphviz
* Try other models (XGBoost, SVM)

---

## 👨‍💻 Author

Rejoy Besra

---
