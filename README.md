# 🛒 Retail Customer Purchase Prediction

## 📌 Project Overview

Retail Customer Purchase Prediction is a Machine Learning project that predicts whether an online shopping session is likely to result in a purchase.

The system analyzes customer browsing behavior and session information and provides a purchase prediction through an interactive Streamlit web application.

The project uses the **Online Shoppers Purchasing Intention Dataset** and applies classification algorithms to predict the target variable `Revenue`.

---

## 🎯 Problem Statement

E-commerce websites generate a large amount of customer browsing and session data.

The objective of this project is to use customer session behavior to predict whether an online shopping session will result in a purchase.

The prediction can help businesses understand customer purchasing patterns and support data-driven customer engagement.

---

## 🎯 Project Objectives

- Analyze online shopping session behavior.
- Perform data preprocessing and cleaning.
- Identify relevant features for purchase prediction.
- Train machine learning classification models.
- Compare Logistic Regression and Random Forest.
- Evaluate models using Accuracy, Precision, Recall and F1 Score.
- Build an interactive Streamlit web application.
- Provide purchase prediction based on customer session details.

---

## 📊 Dataset

### Online Shoppers Purchasing Intention Dataset

The dataset contains **12,330 online shopping sessions** and **18 columns**, including the target variable.

### Target Variable

`Revenue`

- `False` → No Purchase
- `True` → Purchase

### Features

The dataset contains the following features:

- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`
- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`
- `Weekend`

---

## 📈 Dataset Distribution

The target variable is imbalanced.

Approximately:

- **84.53%** sessions → No Purchase
- **15.47%** sessions → Purchase

Because of this class imbalance, multiple evaluation metrics were considered instead of relying only on accuracy.

---

## 🔍 Exploratory Data Analysis

The project includes exploratory analysis of important customer behavior features.

### Key observations

- Product-related page activity differs between purchase and non-purchase sessions.
- Higher `PageValues` are generally associated with sessions that result in purchases.
- Purchase sessions generally show different bounce-rate patterns compared with non-purchase sessions.
- The target variable is imbalanced, with fewer purchase sessions than non-purchase sessions.

These observations show associations in the dataset and do not imply that a particular feature directly causes a purchase.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset structure and data types.
3. Removed missing values.
4. Separated features and target variable.
5. Identified numerical and categorical features.
6. Applied `StandardScaler` to numerical features.
7. Applied `OneHotEncoder` to categorical features.
8. Used `handle_unknown="ignore"` for categorical encoding.
9. Split the dataset into training and testing sets.
10. Used stratified splitting to preserve the target-class distribution.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Feature & Target Selection
   ↓
Categorical Encoding
   ↓
Numerical Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Prediction
```

---

## 🤖 Machine Learning Models

Two classification models were evaluated:

### 1. Logistic Regression

Logistic Regression was used as a classification baseline.

Results:

- Accuracy: **84.10%**
- Precision: **49.13%**
- Recall: **74.35%**
- F1 Score: **59.17%**

---

### 2. Random Forest

Random Forest was used as a tree-based ensemble classification model.

Results:

- Accuracy: **89.58%**
- Precision: **76.16%**
- Recall: **47.64%**
- F1 Score: **58.62%**

The Random Forest model was saved as:

```text
purchase_model.pkl
```

---

## 📊 Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 84.10% | 89.58% |
| Precision | 49.13% | 76.16% |
| Recall | 74.35% | 47.64% |
| F1 Score | 59.17% | 58.62% |

The models show different trade-offs between precision and recall for the purchase class.

---

## 🔢 Random Forest Confusion Matrix

The Random Forest confusion matrix was:

```text
[[2027   57]
 [ 200  182]]
```

Where:

- **True Negative = 2027**
- **False Positive = 57**
- **False Negative = 200**
- **True Positive = 182**

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit web application.

The application allows users to enter customer session information such as:

### Browsing Behaviour

- Administrative Pages
- Administrative Duration
- Informational Pages
- Informational Duration
- Product Related Pages
- Product Related Duration
- Page Value
- Bounce Rate
- Exit Rate
- Special Day

### Session & Device Information

- Operating System
- Browser
- Region
- Traffic Type
- Visitor Type
- Month
- Weekend

The application then generates a purchase prediction.

---

## 🖥️ Application Workflow

```text
User enters session details
          ↓
Streamlit collects input
          ↓
Saved ML pipeline processes the data
          ↓
Random Forest model predicts
          ↓
Purchase / No Purchase result
          ↓
Prediction probability displayed
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Model Storage

- Joblib

### Web Application

- Streamlit

### Data Visualization

- Matplotlib
- Seaborn

### Development Environment

- Visual Studio Code
- Git
- GitHub

---

## 📁 Project Structure

```text
Retail_Purchase_prediction/
│
├── app.py
├── train_model.py
├── check_data.py
├── purchase_model.pkl
├── requirements.txt
├── README.md
│
└── dataset/
    └── online_shoppers_intention.csv
```

---

# ▶️ How to Run

## 1. Create a Virtual Environment

Optional but recommended:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 2. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

## 3. Train the Machine Learning Model

Run:

```bash
python train_model.py
```

This will:

- Load the dataset.
- Clean the data.
- Preprocess numerical and categorical features.
- Split the data into training and testing sets.
- Train the Random Forest model.
- Evaluate the model.
- Save the trained model as:

```text
purchase_model.pkl
```

---

## 4. Run the Streamlit Application

After training the model, run:

```bash
streamlit run app.py
```

The Streamlit application will open in your web browser.

Usually it will be available at:

```text
http://localhost:8501
```

---

# 📦 Requirements

The project uses the following main Python libraries:

```text
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
seaborn
```

The exact versions used for the project are specified in:

```text
requirements.txt
```

---

# 📌 Important Notes

- The model was trained using the Online Shoppers Purchasing Intention Dataset.
- The target variable is `Revenue`.
- `False` represents no purchase and `True` represents purchase.
- The dataset contains imbalanced target classes.
- Therefore, Accuracy, Precision, Recall and F1 Score were considered during evaluation.
- The model predictions depend on the quality and distribution of the training dataset.
- Prediction probability should not be interpreted as a guarantee that a customer will purchase.

---

# 🚀 Future Scope

The project can be improved by:

- Using larger and more diverse e-commerce datasets.
- Adding additional customer behavior features.
- Performing hyperparameter tuning.
- Using cross-validation.
- Testing additional machine learning algorithms.
- Improving class-imbalance handling.
- Adding interactive analytics dashboards.
- Adding model explainability.
- Improving the Streamlit user interface.
- Deploying the application online.

---

# 👩‍💻 Project Information

## Project Title

**Retail Customer Purchase Prediction**

## Domain

**Data Science / Machine Learning**

## Type

**Binary Classification**

## Target

**Online Purchase Prediction**

## Interface

**Streamlit Web Application**

---

## ⭐ Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting online customer purchase behavior.

The workflow covers:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering / Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Streamlit Deployment
```

The project combines Machine Learning with an interactive web interface to demonstrate a practical e-commerce prediction system.