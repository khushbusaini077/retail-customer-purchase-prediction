import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("online_shoppers_intention.csv")

print("Original Dataset Shape:", df.shape)

# Remove missing values
df = df.dropna()

print("After Cleaning:", df.shape)


# ==============================
# 2. Features and Target
# ==============================

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# ==============================
# 3. Define Columns
# ==============================

categorical_cols = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

numerical_cols = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]


# ==============================
# 4. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==================================================
# 6. LOGISTIC REGRESSION
# ==================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)


print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

print("Accuracy :", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred))
print("Recall   :", recall_score(y_test, logistic_pred))
print("F1 Score :", f1_score(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))


# ==================================================
# 7. RANDOM FOREST
# ==================================================

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


print("\n================================")
print("RANDOM FOREST")
print("================================")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==================================================
# 8. Save Random Forest Model
# ==================================================

joblib.dump(rf_model, "purchase_model.pkl")

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")
print("File: purchase_model.pkl")