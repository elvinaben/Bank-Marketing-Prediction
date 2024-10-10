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
    job_value = input_data['job'].iloc[0]  # Get the job value
    if job_value in job_train_unique:
        input_data[f"job_{job_value}"] = 1
    else:
        input_data['job_other'] = 1

    # Encoding marital
    for value in marital_train_unique:
        input_data[f"marital_{value}"] = 0
    marital_value = input_data['marital'].iloc[0]  # Get the marital value
    if marital_value in marital_train_unique:
        input_data[f"marital_{marital_value}"] = 1

    # Encoding education
    for value in education_train_unique:
        input_data[f"education_{value}"] = 0
    input_data['education_other'] = 0
    education_value = input_data['education'].iloc[0]  # Get the education value
    if education_value in education_train_unique:
        input_data[f"education_{education_value}"] = 1
    else:
        input_data['education_other'] = 1

    # Encoding month
    for value in month_train_unique:
        input_data[f"month_{value}"] = 0
    month_value = input_data['month'].iloc[0]  # Get the month value
    if month_value in month_train_unique:
        input_data[f"month_{month_value}"] = 1
    else:
        input_data['month_other'] = 1

    # Encoding day of week
    for value in dow_train_unique:
        input_data[f"day_of_week_{value}"] = 0
    dow_value = input_data['day_of_week'].iloc[0]  # Get the day of week value
    if dow_value in dow_train_unique:
        input_data[f"day_of_week_{dow_value}"] = 1

    # Mapping categorical variables (encode)
    input_data = input_data.apply(lambda col: col.map(mappings['categorical_mappings'][col.name]) if col.name in mappings['categorical_mappings'] else col)
    
    input_data['contact_status'] = input_data['pdays'].apply(lambda x: 1 if x != 999 else 0)

    # Standard scaler
    num_cols = ['age', 'duration', 'campaign', 'previous']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    all_columns = model.feature_names_in_

    # st.write("Feature Names Used in the Model:", all_columns)  # Display the feature names in the app
    # st.write("All Cols Before Deleted:", input_data)

    # Remove unnecessary columns
    for col in all_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[all_columns]

    # st.write("Columns in Input Data After Encoding:", input_data.columns.tolist())
    # st.write("Final Input Data After Preprocessing:", input_data)

    return model.predict(input_data)[0]


st.title("Bank Marketing Prediction 🏦💰")
st.write("Input customer data to determine customer potential for bank offers!")

# User inputs
age = st.number_input("Age", min_value=17, max_value=100)
job = st.selectbox("Job 💼", [format_job(job) for job in job_train_unique])
marital = st.selectbox("Marital Status 💍", [format_capitalized(status) for status in marital_train_unique])
education = st.selectbox("Educational Level 🎓🏫", [format_education(edu) for edu in education_train_unique])
default = st.selectbox("Credit in Default?", ["No", "Yes", "Unknown"])
housing = st.selectbox("Has Housing Loan? 🏠", ["No", "Yes", "Unknown"])
loan = st.selectbox("Has Personal Loan?", ["No", "Yes", "Unknown"])
contact = st.selectbox("Contact Communication Type 📞", ["Cellular", "Telephone"])
month = st.selectbox("Last Contact Month", full_months)
day_of_week = st.selectbox("Last Contact Day of the Week", full_days_of_week)
duration = st.number_input("Last Contact Duration (seconds)", min_value=0)
campaign = st.number_input("Number of Contacts During Campaign", min_value=0)
pdays = st.number_input("Days Since Last Contact", min_value=0, max_value=999)
previous = st.number_input("Previous Contact Count ⏮️", min_value=0)
poutcome = st.selectbox("Outcome of Previous Campaign", ["Failure", "Non-existent", "Success"])

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
        'previous': [previous],
        'poutcome': [poutcome.lower().replace('-', '')]
    })

    prediction = predict(input_data)

    if prediction == 'yes':
        st.markdown(f"<div style='background-color:#d1e7dd;padding:10px;border-radius:5px;'>"
                    f"<strong>Prediction:</strong> This customer is <span style='color:#0f5132;'>a potential target</span> for the bank offer."
                    f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#f8d7da;padding:10px;border-radius:5px;'>"
                    f"<strong>Prediction:</strong> This customer is <span style='color:#842029;'>not a potential target</span> for the bank offer."
                    f"</div>", unsafe_allow_html=True)

# supaya engga pointing ke streamlit python 3.9, runnya di bash pakai:
# python -m streamlit run prediction_streamlit.py

# Cara run normal
# streamlit run prediction_streamlit.py

# Tapi kalau ga bisa pakai yg di atas karena pointing ke python3.9
# What u need to do is : deactivate => uninstall streamlit, then install streamlit again
# Then u can run this : streamlit run prediction_streamlit.py