import streamlit as st
import joblib
import numpy as np
import pandas as pd

import os

# Get absolute path to the current script location
base_path = os.path.dirname(os.path.abspath(__file__))
streamlit_folder = os.path.join(base_path, 'streamlit')

model = joblib.load(os.path.join(streamlit_folder, 'Bagging_dt.pkl'))
mappings = joblib.load(os.path.join(streamlit_folder, 'mappings.pkl'))
scaler = joblib.load(os.path.join(streamlit_folder, 'standard_scaler.pkl'))
job_train_unique = joblib.load(os.path.join(streamlit_folder, 'job_train_unique.pkl'))
education_train_unique = joblib.load(os.path.join(streamlit_folder, 'education_train_unique.pkl'))
marital_train_unique = joblib.load(os.path.join(streamlit_folder, 'marital_train_unique.pkl'))
month_train_unique = joblib.load(os.path.join(streamlit_folder, 'month_train_unique.pkl'))
dow_train_unique = joblib.load(os.path.join(streamlit_folder, 'dow_train_unique.pkl'))


def format_job(job):
    return job.replace('blue-collar', 'Blue collar').title()

def format_education(education):
    return education.replace('.', ' ').title()

def format_capitalized(text):
    return text.capitalize()

# Update month and day of week to full names
full_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
               'August', 'September', 'October', 'November', 'December']

full_days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

def predict(input_data):
    # Encoding job
    for value in job_train_unique:
        input_data[f"job_{value}"] = 0
    input_data['job_other'] = 0
    if input_data['job'].iloc[0] in job_train_unique:
        input_data[f"job_{input_data['job'].iloc[0]}"] = 1
    else:
        input_data['job_other'] = 1

    # Encoding marital
    for value in marital_train_unique:
        input_data[f"marital_{value}"] = 0
    if input_data['marital'].iloc[0] in marital_train_unique:
        input_data[f"marital_{input_data['marital'].iloc[0]}"] = 1

    # Encoding education
    for value in education_train_unique:
        input_data[f"education_{value}"] = 0
    input_data['education_other'] = 0
    if input_data['education'].iloc[0] in education_train_unique:
        input_data[f"education_{input_data['education'].iloc[0]}"] = 1
    else:
        input_data['education_other'] = 1

    # Encoding month
    for value in month_train_unique:
        input_data[f"month_{value}"] = 0
    if input_data['month'].iloc[0] in month_train_unique:
        input_data[f"month_{input_data['month'].iloc[0]}"] = 1

    # Encoding day of week
    for value in dow_train_unique:
        input_data[f"day_of_week_{value}"] = 0
    if input_data['day_of_week'].iloc[0] in dow_train_unique:
        input_data[f"day_of_week_{input_data['day_of_week'].iloc[0]}"] = 1

    # Mapping categorical variables (encode)
    input_data = input_data.apply(lambda col: col.map(mappings['categorical_mappings'][col.name]) if col.name in mappings['categorical_mappings'] else col)

    input_data['contact_status'] = input_data['pdays'].apply(lambda x: 1 if x != 999 else 0)

    # Standard scaler
    num_cols = ['age', 'duration', 'campaign', 'previous']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    all_columns = model.feature_names_in_
    for col in all_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[all_columns]

    prediction = model.predict(input_data)

    return prediction[0]


st.title("Bank Marketing Prediction")
st.write("Input customer data to determine customer potential for bank offers!")

# User inputs
age = st.number_input("Age", min_value=17, max_value=100)
job = st.selectbox("Job", [format_job(job) for job in job_train_unique])
marital = st.selectbox("Marital Status", [format_capitalized(status) for status in marital_train_unique])
education = st.selectbox("Education", [format_education(edu) for edu in education_train_unique])
default = st.selectbox("Default", ["No", "Yes", "Unknown"])
housing = st.selectbox("Housing", ["No", "Yes", "Unknown"])
loan = st.selectbox("Loan", ["No", "Yes", "Unknown"])
contact = st.selectbox("Contact", ["Cellular", "Telephone"])
month = st.selectbox("Month", full_months)
day_of_week = st.selectbox("Day of Week", full_days_of_week)
duration = st.number_input("Duration", min_value=0)
campaign = st.number_input("Campaign", min_value=0)
pdays = st.number_input("Pdays", min_value=0, max_value=999)
previous = st.number_input("Previous", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job.lower()],  
        'marital': [marital.lower()],
        'education': [education.lower().replace(' ', '.')],  
        'default': [default.lower()],
        'housing': [housing.lower()],
        'loan': [loan.lower()],
        'contact': [contact.lower()],
        'month': [month.lower()],
        'day_of_week': [day_of_week[:3].lower()],  
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous]
    })

    prediction = predict(input_data)
    st.write(f"Prediction: This customer is {'a potential target' if prediction == 1 else 'not a potential target'} for the bank offer.")
