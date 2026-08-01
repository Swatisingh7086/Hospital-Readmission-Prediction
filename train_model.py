import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "diabetic_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Dataset...")

df = pd.read_csv(DATA_PATH)

print("Original Shape :", df.shape)

drop_columns = [
    "weight",
    "payer_code",
    "medical_specialty"
]

df.drop(columns=drop_columns, inplace=True)

expired = [11, 13, 14, 19, 20, 21]

df = df[
    ~df["discharge_disposition_id"].isin(expired)
]

df["readmitted_30d"] = (
    df["readmitted"] == "<30"
).astype(int)

df.drop(columns=["readmitted"], inplace=True)


def convert_age(age):
    age = age.replace("[", "").replace(")", "")
    low = int(age.split("-")[0])
    high = int(age.split("-")[1])
    return (low + high) / 2


df["age_numeric"] = df["age"].apply(convert_age)

diagnosis_columns = [
    "diag_1",
    "diag_2",
    "diag_3"
]

df["num_diag_populated"] = (
    df[diagnosis_columns]
    .apply(lambda x: (x != "?").sum(), axis=1)
)

drug_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

df["num_med_changes"] = (
    (df[drug_columns] == "Up").sum(axis=1)
    +
    (df[drug_columns] == "Down").sum(axis=1)
)

df["total_prior_visits"] = (
    df["number_outpatient"]
    +
    df["number_emergency"]
    +
    df["number_inpatient"]
)

df.drop(
    columns=[
        "encounter_id",
        "patient_nbr",
        "diag_1",
        "diag_2",
        "diag_3",
        "age"
    ],
    inplace=True
)

df.drop(columns=drug_columns, inplace=True)

df.replace("?", "Unknown", inplace=True)

categorical_columns = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed"
]

for col in categorical_columns:
    df[col] = df[col].astype(str)

print("\nDataset Preview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("\nEncoded Shape :", df.shape)

X = df.drop("readmitted_30d", axis=1)
y = df["readmitted_30d"]

feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nModel Performance")
print("-"*50)
print("Accuracy :", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall   :", recall_score(y_test,y_pred))
print("F1 Score :", f1_score(y_test,y_pred))
print("ROC AUC  :", roc_auc_score(y_test,y_prob))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "rf_balanced_model.joblib"
    )
)

joblib.dump(
    feature_columns,
    os.path.join(
        MODEL_DIR,
        "feature_columns.joblib"
    )
)

print("\nModel Saved Successfully")

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Readmission Distribution")
plt.xlabel("Readmitted Within 30 Days")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["time_in_hospital"], bins=14, kde=True)
plt.title("Time in Hospital")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["num_lab_procedures"], bins=20, kde=True)
plt.title("Lab Procedures")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["num_medications"], bins=20, kde=True)
plt.title("Number of Medications")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["age_numeric"], bins=10, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["time_in_hospital"])
plt.title("Time in Hospital Boxplot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["num_medications"])
plt.title("Medication Boxplot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No","Yes"],
    yticklabels=["No","Yes"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(20)

plt.figure(figsize=(10,8))
importance.plot(kind="barh")
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test,y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f"AUC={roc_auc_score(y_test,y_prob):.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
df["number_diagnoses"].value_counts().sort_index().plot(kind="bar")
plt.title("Number of Diagnoses")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
df["gender_Male"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
df["diabetesMed_Yes"].value_counts().plot(kind="bar")
plt.title("Diabetes Medication")
plt.tight_layout()
plt.show()

print("\nTraining Completed Successfully")
