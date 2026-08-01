import pandas as pd
from utils import predict_risk


def make_prediction(user_data):

    return predict_risk(user_data)


def display_result(user_data):

    result = make_prediction(user_data)

    print("\n" + "=" * 50)
    print("HOSPITAL READMISSION PREDICTION")
    print("=" * 50)

    print("\nPatient Details")

    for key, value in user_data.items():
        print(f"{key:25}: {value}")

    print("\nPrediction Result")
    print("-" * 50)

    print(f"Risk Level      : {result['risk']}")
    print(f"Probability     : {result['probability']:.2%}")
    print(f"Recommendation  : {result['recommendation']}")

    print("=" * 50)

    return result


if __name__ == "__main__":

    sample_patient = {

        "age_numeric": 60,
        "race": "Caucasian",
        "gender": "Male",
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "7",
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 8,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "num_diag_populated": 3,
        "num_med_changes": 0,
        "total_prior_visits": 1

    }

    display_result(sample_patient)