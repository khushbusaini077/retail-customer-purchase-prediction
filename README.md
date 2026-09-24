# 🛒 Retail Customer Purchase Prediction

## 📌 Project Overview

Retail Customer Purchase Prediction is a machine learning based system that predicts whether an online shopping session is likely to result in a purchase.

The system analyzes customer browsing behavior and session characteristics and provides:

- Purchase / No Purchase prediction
- Purchase probability
- Model-based prediction using Random Forest

The application is built using Python, Scikit-learn and Streamlit.

---

## 🎯 Problem Statement

E-commerce platforms receive a large number of customer sessions every day.

The objective of this project is to predict whether a customer session is likely to result in a purchase based on available browsing and visitor behavior.

This can help businesses identify sessions with higher purchase likelihood and support data-driven decision making.

---

## 📊 Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** Kaggle

**Original Source:** UCI Machine Learning Repository

The dataset contains 12,330 online shopping sessions and 18 columns.

### Target Variable

`Revenue`

- `True` → Purchase
- `False` → No Purchase

### Main Features

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

---

## 🔄 Machine Learning Workflow

```text
Dataset
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
Random Forest Model
   ↓
Streamlit Prediction App