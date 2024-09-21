import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and artifacts
model = joblib.load('Bagging_dt.pkl')
mappings = joblib.load('mappings.pkl')
scaler = joblib.load('standard_scaler.pkl')
job_train_unique = joblib.load('job_train_unique.pkl')
education_train_unique = joblib.load('education_train_unique.pkl')
marital_train_unique = joblib.load('marital_train_unique.pkl')
month_train_unique = joblib.load('month_train_unique.pkl')
dow_train_unique = joblib.load('dow_train_unique.pkl')

def predict(input_data):
    # Encoding job
    for value in job_train_unique:
        input_data[f"job_{value}"] = 0
    input_data['job_other'] = 0
    if input_data.job in job_train_unique:
        input_data[f"job_{input_data.job}"] = 1
    else:
        input_data['job_other'] = 1

    # Encoding marital
    for value in marital_train_unique:
        input_data[f"marital_{value}"] = 0
    if input_data.marital in marital_train_unique:
        input_data[f"marital_{input_data.marital}"] = 1

    # Encoding education
    for value in education_train_unique:
        input_data[f"education_{value}"] = 0
    input_data['education_other'] = 0
    if input_data.education in education_train_unique:
        input_data[f"education_{input_data.education}"] = 1
    else:
        input_data['education_other'] = 1

    # Encoding month
    for value in month_train_unique:
        input_data[f"month_{value}"] = 0
    if input_data.month in month_train_unique:
        input_data[f"month_{input_data.month}"] = 1

    # Encoding day of week
    for value in dow_train_unique:
        input_data[f"day_of_week_{value}"] = 0
    if input_data.day_of_week in dow_train_unique:
        input_data[f"day_of_week_{input_data.day_of_week}"] = 1

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
job = st.selectbox("Job", job_train_unique)
marital = st.selectbox("Marital Status", marital_train_unique)
education = st.selectbox("Education", education_train_unique)
default = st.selectbox("Default", ["no", "yes", "unknown"])
housing = st.selectbox("Housing", ["no", "yes", "unknown"])
loan = st.selectbox("Loan", ["no", "yes", "unknown"])
contact = st.selectbox("Contact", ["cellular", "telephone"])
month = st.selectbox("Month", month_train_unique)
day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
duration = st.number_input("Duration", min_value=0)
campaign = st.number_input("Campaign", min_value=0)
pdays = st.number_input("Pdays", min_value=0, max_value=999)
previous = st.number_input("Previous", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous]
    })

    prediction = predict(input_data)
    st.write(f"Prediction: This customer is {'a potential target' if prediction == 1 else 'not a potential target'} for the bank offer.")
