import requests
import streamlit as st

API_URL = st.sidebar.text_input("API base URL", "https://kashan425-salary-prediction-backend.hf.space")
WORKCLASS_OPTIONS = [
    "Federal-gov",
    "Local-gov",
    "Never-worked",
    "Private",
    "Self-emp-inc",
    "Self-emp-not-inc",
    "State-gov",
    "Without-pay",
]
EDUCATION_OPTIONS = [
    "10th",
    "11th",
    "12th",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "Assoc-acdm",
    "Assoc-voc",
    "Bachelors",
    "Doctorate",
    "HS-grad",
    "Masters",
    "Preschool",
    "Prof-school",
    "Some-college",
]
MARITAL_STATUS_OPTIONS = [
    "Divorced",
    "Married-AF-spouse",
    "Married-civ-spouse",
    "Married-spouse-absent",
    "Never-married",
    "Separated",
    "Widowed",
]
OCCUPATION_OPTIONS = [
    "Adm-clerical",
    "Armed-Forces",
    "Craft-repair",
    "Exec-managerial",
    "Farming-fishing",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Other-service",
    "Priv-house-serv",
    "Prof-specialty",
    "Protective-serv",
    "Sales",
    "Tech-support",
    "Transport-moving",
]
RELATIONSHIP_OPTIONS = [
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Own-child",
    "Unmarried",
    "Wife",
]
RACE_OPTIONS = [
    "Amer-Indian-Eskimo",
    "Asian-Pac-Islander",
    "Black",
    "Other",
    "White",
]
GENDER_OPTIONS = ["Female", "Male"]
NATIVE_COUNTRY_OPTIONS = [
    "Cambodia",
    "Canada",
    "China",
    "Columbia",
    "Cuba",
    "Dominican-Republic",
    "Ecuador",
    "El-Salvador",
    "England",
    "France",
    "Germany",
    "Greece",
    "Guatemala",
    "Haiti",
    "Holand-Netherlands",
    "Honduras",
    "Hong",
    "Hungary",
    "India",
    "Iran",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Laos",
    "Mexico",
    "Nicaragua",
    "Outlying-US(Guam-USVI-etc)",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Puerto-Rico",
    "Scotland",
    "South",
    "Taiwan",
    "Thailand",
    "Trinadad&Tobago",
    "United-States",
    "Vietnam",
    "Yugoslavia",
]


def prediction_form() -> None:
    st.sidebar.header("Income prediction")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=35)
    fnlwgt = st.sidebar.number_input("Final weight (fnlwgt)", min_value=0, value=123000)
    workclass = st.sidebar.selectbox("Workclass", WORKCLASS_OPTIONS)
    education = st.sidebar.selectbox("Education", EDUCATION_OPTIONS)
    educational_num = st.sidebar.number_input("Education-num", min_value=0, max_value=20, value=10)
    marital_status = st.sidebar.selectbox("Marital status", MARITAL_STATUS_OPTIONS)
    occupation = st.sidebar.selectbox("Occupation", OCCUPATION_OPTIONS)
    relationship = st.sidebar.selectbox("Relationship", RELATIONSHIP_OPTIONS)
    race = st.sidebar.selectbox("Race", RACE_OPTIONS)
    gender = st.sidebar.selectbox("Gender", GENDER_OPTIONS)
    capital_gain = st.sidebar.number_input("Capital gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital loss", min_value=0, value=0)
    hours_per_week = st.sidebar.number_input("Hours per week", min_value=0, max_value=168, value=40)
    native_country = st.sidebar.selectbox("Native country", NATIVE_COUNTRY_OPTIONS)

    if st.sidebar.button("Predict income"):
        payload = {
            "age": age,
            "fnlwgt": fnlwgt,
            "workclass": workclass,
            "education": education,
            "educational-num": educational_num,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": native_country,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.success(f"Predicted income: {result['prediction']}")
            if result.get("confidence") is not None:
                st.write(f"Confidence: {result['confidence']:.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def run() -> None:
    st.title("Adult Income Prediction")
    st.write(
        "Use the sidebar to choose demographic and income-related features, then click Predict income."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### API settings")
    st.sidebar.write("Use FastAPI URL for prediction requests")
    st.sidebar.write("Example: http://localhost:8000")
    prediction_form()


if __name__ == "__main__":
    run()
