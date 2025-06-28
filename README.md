# 💳 Credit Card Fraud Detection using Machine Learning

> A machine learning project aimed at detecting fraudulent credit card transactions using supervised classification techniques on imbalanced datasets.

## 📌 Problem Statement

Credit card fraud causes significant financial losses each year. Given a dataset of real transactions labeled as fraudulent or legitimate, the goal is to build a machine learning model that can effectively detect fraud, despite the severe class imbalance.

---

## 🚀 Project Overview

This project uses a highly imbalanced dataset to classify transactions as **fraudulent** or **legitimate**. We apply data preprocessing, undersampling, and train multiple ML models including Logistic Regression and Decision Trees to evaluate performance using appropriate metrics like Precision, Recall, and F1-score.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**:
  - `Time`, `Amount` (raw)
  - `V1` to `V28` (anonymized PCA components)
  - `Class`: 1 = Fraud, 0 = Legitimate

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Dropped duplicate rows
- Normalized `Amount` column using `StandardScaler`
- Dropped `Time` column (not predictive)

### 2. Handling Imbalanced Data
- Applied **undersampling**: Equalized samples of fraud and non-fraud classes
- Created a new balanced dataset with ~946 samples (473 per class)

### 3. Train-Test Split
- Used `train_test_split` with `test_size=0.2` and `random_state=42`

### 4. Model Training
- Logistic Regression
- Decision Tree Classifier

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ✅        | ✅         | ✅     | ✅        |
| Decision Tree      | ✅        | ✅         | ✅     | ✅        |

> Focus was on **Recall** and **F1-score** due to class imbalance.

---

## 📦 Technologies Used

- **Python**
- **Pandas, NumPy** – Data handling
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, Modeling, Evaluation

---

## 🧠 Why Each Step Was Done

- **StandardScaler**: Normalize skewed `Amount` for better model performance
- **Undersampling**: Balance dataset to avoid biased models
- **Logistic Regression**: Simple baseline model for classification
- **Decision Tree**: To capture non-linear relationships
- **Evaluation Metrics**: Focused on Recall & F1-score due to fraud class importance

---

## 📌 Future Improvements

- Use **SMOTE** or **ADASYN** for oversampling instead of undersampling
- Try **Ensemble models** like Random Forest or XGBoost
- Implement **PCA** for dimensionality reduction if overfitting occurs
- Deploy model via Flask or Streamlit

---

## 📁 Project Status

✅ Completed core implementation  
🚧 Open for improvements and deployment  
📈 Performance depends on train-test split randomness

---

## 🤝 Contributions

This project was implemented as part of a portfolio targeted toward roles in Data Science, ML, and FinTech analytics.
