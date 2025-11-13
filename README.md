# 🚢 Titanic Survival Prediction — Logistic Regression in PyTorch

This project builds a **Logistic Regression model from scratch using PyTorch** to predict whether a passenger survived the Titanic disaster.  
It includes:

- Full preprocessing pipeline  
- Feature engineering (Ticket/Cabin parsing)  
- Dataset normalization  
- Binary classification using Logistic Regression  
- Custom training loop using `BCEWithLogitsLoss`  
- Testing on actual Titanic training data  
- Final accuracy evaluation  

This repository is a clean, easy-to-understand example of traditional ML models implemented with **deep-learning tools (PyTorch)**.

---

## 📁 Dataset Description

The project uses the Kaggle Titanic dataset:

- **test.csv** → model training in this project  
- **gender_submission.csv** → ground-truth labels for test.csv  
- **train.csv** → used later to evaluate the trained model

### Input Features Used:
| Feature | Description |
|--------|-------------|
| Pclass | Ticket class (1, 2, 3) |
| Sex | Male/Female mapped to 0/1 |
| Age | Filled missing values |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Ticket | Last numeric part extracted |
| Fare | Passenger fare |
| Cabin | Categorical flag (0 = None, 1 = Has cabin) |
| Embarked | Encoded as C=1, Q=2, S=3 |

---

## 🧹 Data Preprocessing

### 1️⃣ Drop unused columns
```python
df = df.drop(["PassengerId", "Name"], axis=1)
