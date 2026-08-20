import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("student_dropout_risk_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Nigeria Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Nigeria Student Dropout Risk Predictor")

st.write(
    "Enter student information below to estimate the student's "
    "dropout-risk category using a machine-learning model."
)

# Student information
age = st.number_input(
    "Age",
    min_value=10,
    max_value=30,
    value=17
)

attendance_rate = st.number_input(
    "Attendance Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=68.0
)

average_score = st.number_input(
    "Average Score (%)",
    min_value=0.0,
    max_value=100.0,
    value=55.0
)

previous_failures = st.number_input(
    "Previous Failures",
    min_value=0,
    max_value=10,
    value=2
)

distance_to_school_km = st.number_input(
    "Distance to School (km)",
    min_value=0.0,
    max_value=100.0,
    value=8.5
)

dependents_in_household = st.number_input(
    "Dependents in Household",
    min_value=0,
    max_value=20,
    value=6
)

study_hours_per_week = st.number_input(
    "Study Hours per Week",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

state = st.selectbox(
    "State",
    [
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi",
        "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
        "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe",
        "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
        "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
        "Niger", "Ogun", "Ondo", "Osun", "Oyo",
        "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe",
        "Zamfara"
    ]
)

location = st.selectbox(
    "Location",
    ["Urban", "Rural"]
)

school_type = st.selectbox(
    "School Type",
    ["Public", "Private"]
)

household_income_level = st.selectbox(
    "Household Income Level",
    ["Low", "Lower-Middle", "Middle", "Upper-Middle"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["Yes", "No"]
)

learning_material_access = st.selectbox(
    "Learning Material Access",
    ["Adequate", "Limited"]
)

school_fees_status = st.selectbox(
    "School Fees Status",
    ["Paid", "Behind"]
)

scholarship_support = st.selectbox(
    "Scholarship Support",
    ["Yes", "No"]
)

parent_education_level = st.selectbox(
    "Parent Education Level",
    ["Primary", "Secondary", "Tertiary"]
)

# Prediction
if st.button("Predict Dropout Risk"):

    student = pd.DataFrame([{
        "age": age,
        "attendance_rate": attendance_rate,
        "average_score": average_score,
        "previous_failures": previous_failures,
        "distance_to_school_km": distance_to_school_km,
        "dependents_in_household": dependents_in_household,
        "study_hours_per_week": study_hours_per_week,
        "gender": gender,
        "state": state,
        "location": location,
        "school_type": school_type,
        "household_income_level": household_income_level,
        "internet_access": internet_access,
        "learning_material_access": learning_material_access,
        "school_fees_status": school_fees_status,
        "scholarship_support": scholarship_support,
        "parent_education_level": parent_education_level
    }])

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    labels = ["Low", "Medium", "High"]
    predicted_risk = labels[prediction]

    st.subheader("Prediction Result")

    if predicted_risk == "High":
        st.error(f"Predicted Dropout Risk: {predicted_risk}")
    elif predicted_risk == "Medium":
        st.warning(f"Predicted Dropout Risk: {predicted_risk}")
    else:
        st.success(f"Predicted Dropout Risk: {predicted_risk}")

    st.write("### Model Probabilities")

    st.write(f"Low Risk: {probabilities[0] * 100:.2f}%")
    st.write(f"Medium Risk: {probabilities[1] * 100:.2f}%")
    st.write(f"High Risk: {probabilities[2] * 100:.2f}%")

st.divider()

st.caption(
    "This tool is an AI/ML prototype developed for academic purposes. "
    "Predictions are model estimates and should not be treated as definitive "
    "evidence that a student will drop out."
)
