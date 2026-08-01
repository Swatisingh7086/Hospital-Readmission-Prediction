import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "rf_balanced_model.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "feature_columns.joblib"
)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def preprocess_input(user_data):

    df = pd.DataFrame([user_data])

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

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


def predict_readmission(user_data):

    processed_data = preprocess_input(user_data)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    return int(prediction), float(probability)


def predict_risk(user_data):

    prediction, probability = predict_readmission(user_data)

    if prediction == 1:
        risk = "High Risk of Readmission"
        recommendation = (
            "Patient should receive close monitoring, "
            "follow-up care, and medication review."
        )
    else:
        risk = "Low Risk of Readmission"
        recommendation = (
            "Continue routine care and regular follow-up."
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "risk": risk,
        "recommendation": recommendation
    }


def load_model():
    return model


def load_feature_columns():
    return feature_columns


if __name__ == "__main__":

    sample_patient = {
        "age_numeric": 65,
        "race": "Caucasian",
        "gender": "Male",
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "7",
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 1,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 8,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "num_diag_populated": 3,
        "num_med_changes": 0,
        "total_prior_visits": 2
    }

    result = predict_risk(sample_patient)

    print("\nPrediction")
    print("-" * 40)
    print("Risk Level :", result["risk"])
    print("Probability :", f"{result['probability']:.2%}")
    print("Recommendation :", result["recommendation"])