import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('best_salary_model.pkl')

# Title
st.title("Employee Salary Prediction App")

# Input form
st.subheader("Enter Employee Details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
experience = st.slider("Years of Experience", 0, 50, 5)
job_titles = ['Data Analyst', 'Data Scientist', 'HR', 'Manager', 'Others', 'Sales Executive', 'Software Engineer']  # Use actual job title columns from training
job_title = st.selectbox("Job Title", job_titles)

# Preprocess input
gender = 1 if gender == 'Male' else 0
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education = education_map[education]

# Create dummy variables for job titles
job_title_data = {f'Job Title_{jt}': 0 for jt in job_titles if jt != 'Others'}
if job_title != 'Others':
    job_title_data[f'Job Title_{job_title}'] = 1

# Final input array
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Education Level': education,
    'Years of Experience': experience,
    **job_title_data
}])

# Align columns with training data
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model.feature_names_in_]

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ₹{prediction:,.2f}")
