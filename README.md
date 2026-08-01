# 🏥 Hospital Readmission Prediction

A Machine Learning and Streamlit project that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

---

# Project Overview

Hospital readmissions increase healthcare costs and indicate potential issues in patient care. This project uses Machine Learning to predict whether a patient will be readmitted within 30 days based on demographic, medical, and hospital information.

---

# Features

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Random Forest Classification
- Interactive Streamlit Dashboard
- Readmission Prediction
- Probability Score
- Model Performance
- Interactive Charts

---

# Dataset

Dataset Used:

Diabetes 130-US Hospitals Dataset

Files

- diabetic_data.csv
- IDS_mapping.csv

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# Machine Learning Workflow

1. Load Dataset

2. Data Cleaning

3. Remove Missing Values

4. Feature Engineering

5. One-Hot Encoding

6. Train-Test Split

7. Random Forest Training

8. Model Evaluation

9. Save Model

10. Deploy using Streamlit

---

# Folder Structure

```
Hospital_Readmission_Prediction/
│
├── app.py
├── train_model.py
├── utils.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Dataset/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
│
├── model/
│   ├── rf_balanced_model.joblib
│   └── feature_columns.joblib
│
└── Images/
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd Hospital_Readmission_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Machine Learning Model

Algorithm Used

Random Forest Classifier

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

# Streamlit Pages

- Home
- Dataset
- Visualizations
- Prediction
- Model Performance
- About

---

# Future Improvements

- Hyperparameter Tuning
- XGBoost Implementation
- SHAP Explainability
- Patient Risk Dashboard
- Cloud Deployment
- Feature Importance Dashboard

---

# Developer

Swati Singh

---

# License

This project is developed for educational purposes.