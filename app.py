import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import predict_readmission

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "diabetic_data.csv"
)

df = pd.read_csv(DATA_PATH)

st.sidebar.title("🏥 Hospital Readmission")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Visualizations",
        "Prediction",
        "Model Performance",
        "About"
    ]
)

if page == "Home":

    st.title("🏥 Hospital Readmission Prediction System")

    st.markdown("""
## Welcome

This application predicts whether a diabetic patient is likely to be readmitted to the hospital within **30 days** using Machine Learning.

### Features

- Dataset Exploration
- Interactive Visualizations
- Readmission Prediction
- Random Forest Model
- Performance Analysis
""")

    st.success("Select a page from the sidebar.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Features",
        df.shape[1]
    )

    col3.metric(
        "Readmission Records",
        df["readmitted"].count()
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns.tolist())

elif page == "Dataset":

    st.title("📂 Dataset Overview")

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Data Types")

    st.dataframe(
        pd.DataFrame(
            df.dtypes,
            columns=["Data Type"]
        ),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = (
        (df == "?").sum()
        +
        df.isnull().sum()
    )

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

elif page == "Visualizations":

    st.title("📊 Exploratory Data Analysis")

    chart = st.selectbox(
        "Choose Graph",
        [
            "Target Distribution",
            "Age Distribution",
            "Gender Distribution",
            "Race Distribution",
            "Time In Hospital",
            "Lab Procedures",
            "Medication Distribution",
            "Diagnosis Distribution",
            "Correlation Heatmap"
        ]
    )

    if chart == "Target Distribution":

        fig, ax = plt.subplots(figsize=(7,5))

        sns.countplot(
            x="readmitted",
            data=df,
            ax=ax
        )

        ax.set_title("Readmission Distribution")

        st.pyplot(fig)

    elif chart == "Age Distribution":

        fig, ax = plt.subplots(figsize=(10,5))

        sns.countplot(
            data=df,
            x="age",
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    elif chart == "Gender Distribution":

        fig, ax = plt.subplots(figsize=(6,5))

        sns.countplot(
            data=df,
            x="gender",
            ax=ax
        )

        st.pyplot(fig)

    elif chart == "Race Distribution":

        fig, ax = plt.subplots(figsize=(8,5))

        sns.countplot(
            data=df,
            x="race",
            order=df["race"].value_counts().index,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    elif chart == "Time In Hospital":

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            data=df,
            x="time_in_hospital",
            bins=14,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)
    elif chart == "Lab Procedures":

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            data=df,
            x="num_lab_procedures",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_title("Lab Procedures")

        st.pyplot(fig)

    elif chart == "Medication Distribution":

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            data=df,
            x="num_medications",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_title("Number of Medications")

        st.pyplot(fig)

    elif chart == "Diagnosis Distribution":

        fig, ax = plt.subplots(figsize=(8,5))

        sns.countplot(
            data=df,
            x="number_diagnoses",
            ax=ax
        )

        plt.xticks(rotation=90)

        ax.set_title("Number of Diagnoses")

        st.pyplot(fig)

    elif chart == "Correlation Heatmap":

        numeric = df.select_dtypes(include=np.number)

        fig, ax = plt.subplots(figsize=(12,10))

        sns.heatmap(
            numeric.corr(),
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)

elif page == "Prediction":

    st.title("🤖 Hospital Readmission Prediction")

    st.write("Enter the patient details below.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            0,
            100,
            50
        )

        race = st.selectbox(
            "Race",
            sorted(df["race"].astype(str).unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].astype(str).unique())
        )

        admission_type = st.selectbox(
            "Admission Type",
            sorted(df["admission_type_id"].astype(str).unique())
        )

        discharge = st.selectbox(
            "Discharge Disposition",
            sorted(df["discharge_disposition_id"].astype(str).unique())
        )

        admission_source = st.selectbox(
            "Admission Source",
            sorted(df["admission_source_id"].astype(str).unique())
        )

        max_glucose = st.selectbox(
            "Maximum Glucose Serum",
            sorted(df["max_glu_serum"].astype(str).unique())
        )

        a1c = st.selectbox(
            "A1C Result",
            sorted(df["A1Cresult"].astype(str).unique())
        )

        change = st.selectbox(
            "Medication Change",
            sorted(df["change"].astype(str).unique())
        )

        diabetes_med = st.selectbox(
            "Diabetes Medication",
            sorted(df["diabetesMed"].astype(str).unique())
        )

    with col2:

        time_in_hospital = st.number_input(
            "Time In Hospital",
            1,
            14,
            5
        )

        num_lab = st.number_input(
            "Lab Procedures",
            1,
            150,
            40
        )

        num_proc = st.number_input(
            "Procedures",
            0,
            20,
            1
        )

        num_med = st.number_input(
            "Medications",
            1,
            100,
            15
        )

        outpatient = st.number_input(
            "Outpatient Visits",
            0,
            50,
            0
        )

        emergency = st.number_input(
            "Emergency Visits",
            0,
            50,
            0
        )

        inpatient = st.number_input(
            "Inpatient Visits",
            0,
            50,
            0
        )

        diagnoses = st.number_input(
            "Number of Diagnoses",
            1,
            20,
            5
        )

    total_visits = (
        outpatient +
        emergency +
        inpatient
    )

    user_data = {

        "age_numeric": age,
        "race": race,
        "gender": gender,
        "admission_type_id": admission_type,
        "discharge_disposition_id": discharge,
        "admission_source_id": admission_source,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab,
        "num_procedures": num_proc,
        "num_medications": num_med,
        "number_outpatient": outpatient,
        "number_emergency": emergency,
        "number_inpatient": inpatient,
        "number_diagnoses": diagnoses,
        "max_glu_serum": max_glucose,
        "A1Cresult": a1c,
        "change": change,
        "diabetesMed": diabetes_med,
        "num_diag_populated": 3,
        "num_med_changes": 0,
        "total_prior_visits": total_visits
    }

    if st.button(
        "Predict Readmission",
        use_container_width=True
    ):

        prediction, probability = predict_readmission(user_data)

        st.divider()

        if prediction == 1:

            st.error("⚠ High Risk of Hospital Readmission")

        else:

            st.success("✅ Low Risk of Hospital Readmission")

        st.metric(
            "Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

        summary = pd.DataFrame(
            user_data.items(),
            columns=[
                "Feature",
                "Value"
            ]
        )

        st.subheader("Patient Summary")

        st.dataframe(
            summary,
            use_container_width=True
        )
elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.subheader("Random Forest Classifier")

    performance = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Value": [
                "Check train_model.py output",
                "Check train_model.py output",
                "Check train_model.py output",
                "Check train_model.py output",
                "Check train_model.py output"
            ]
        }
    )

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.subheader("Machine Learning Workflow")

    workflow = [
        "Load Dataset",
        "Data Cleaning",
        "Feature Engineering",
        "Handle Missing Values",
        "One-Hot Encoding",
        "Train-Test Split",
        "Random Forest Training",
        "Model Evaluation",
        "Save Model",
        "Deploy with Streamlit"
    ]

    for step in workflow:
        st.write("✅", step)

    st.subheader("Feature Summary")

    feature_info = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Time In Hospital",
                "Lab Procedures",
                "Medications",
                "Diagnoses",
                "Previous Visits"
            ],
            "Description": [
                "Patient Age",
                "Length of Stay",
                "Number of Lab Tests",
                "Number of Medicines",
                "Diagnosis Count",
                "Outpatient + Emergency + Inpatient Visits"
            ]
        }
    )

    st.dataframe(
        feature_info,
        use_container_width=True
    )

elif page == "About":

    st.title("ℹ️ About Project")

    st.markdown(
        """
# Hospital Readmission Prediction

This project predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days using Machine Learning.

## Dataset

Diabetes 130-US Hospitals Dataset

## Machine Learning Algorithm

Random Forest Classifier

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

## Features

- Dataset Exploration
- Interactive Visualizations
- Hospital Readmission Prediction
- Probability Score
- Performance Evaluation
- Feature Engineering
- Streamlit Deployment

## Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Encoding
5. Model Training
6. Model Evaluation
7. Prediction
8. Deployment
"""
    )

    st.subheader("Developer")

    st.write("Swati Singh")

    st.subheader("Project")

    st.write("Hospital Readmission Prediction using Machine Learning")

st.sidebar.markdown("---")

st.sidebar.success("Hospital Readmission Prediction")

st.sidebar.write("Developed using Streamlit")

st.sidebar.write("Random Forest Classifier")

st.sidebar.write("Version 1.0")
