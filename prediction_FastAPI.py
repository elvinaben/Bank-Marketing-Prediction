from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

model = joblib.load('Bagging_dt.pkl')
mappings = joblib.load('mappings.pkl')
scaler = joblib.load('standard_scaler.pkl')
job_train_unique = joblib.load('job_train_unique.pkl')
education_train_unique = joblib.load('education_train_unique.pkl')
marital_train_unique = joblib.load('marital_train_unique.pkl')
month_train_unique = joblib.load('month_train_unique.pkl')
dow_train_unique = joblib.load('dow_train_unique.pkl')


class Data(BaseModel):
    age: int = Field(..., gt=16, le=100, description="Age must be between 17 and 100")
    job: str = Field(..., description="Job must be a string")
    marital: str = Field(..., description="Marital status must be a string") 
    education: str = Field(..., description="Education must be a string")
    default: str = Field(..., description="Default must be 'no', 'yes', or 'unknown'")
    housing: str = Field(..., description="Housing must be 'no', 'yes', or 'unknown'")
    loan: str = Field(..., description="Loan must be 'no', 'yes', or 'unknown'")
    contact: str = Field(..., description="Contact must be 'cellular' or 'telephone'")
    month: str = Field(..., description="Month must be 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', or 'dec'")
    day_of_week: str = Field(..., description="Day must be 'mon', 'tue', 'wed', 'thu', or 'fri'")
    duration: int = Field(..., ge=0, description="Duration must be a non-negative integer")
    campaign: int = Field(..., ge=0, description="Campaign must be a non-negative integer")
    pdays: int = Field(..., ge=0, le=999, description="Pdays must be a non-negative integer or 999 if never contacted")
    previous: int = Field(..., ge=0, description="Previous must be a non-connegative integer")
    poutcome: str = Field(..., description="Poutcome must be 'failure', 'nonexistent', or 'success'")

    # Validator to prevent empty string inputs
    @validator('default', 'housing', 'loan', 'contact', 'poutcome', 'month', 'day_of_week', 'education', 'job', 'marital')
    def check_not_empty(cls, v):
        if not v:
            raise ValueError("Field cannot be empty")
        return v

    # Validation to prevent the education and job inputs from remaining as "string"
    @validator('education', 'job')
    def check_not_empty_string(cls, v):
        if v in {'string'}:  
            raise ValueError("Field must be filled in accordance")
        return v 

    # Education input must only contain lowercase letters a-z, digits, and dots
    @validator('education')
    def education_validator(cls, v):
        if any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789.' for char in v.lower()):
            raise ValueError("Education must only contain lowercase letters a-z, digits, and dots. Use dot for space! (e.g. high.school)")
        return v

    # Job input must only contain lowercase letters a-z, dots, and dashes
    @validator('job')
    def job_validator(cls, v):
        if any(char not in 'abcdefghijklmnopqrstuvwxyz.-' for char in v.lower()):
            raise ValueError("Job must only contain lowercase letters a-z, dash, and dots. Use dot for space! (e.g. private.chef)")
        return v

    # To handle inputs that end with a non-alphabetic character, such as 'admin.'
    @validator('education', 'job')
    def trim_invalid_ending_char(cls, v):
        while len(v) > 0 and v[-1] not in 'abcdefghijklmnopqrstuvwxyz':
            v = v[:-1]
        return v

    # Input is validated to be one of "default," "housing," or "loan."
    @validator('default', 'housing', 'loan')
    def check_boolean_strings(cls, v):
        if v not in {'no', 'yes', 'unknown'}:
            raise ValueError("Must be 'no', 'yes', or 'unknown'")
        return v

    # Input is validated to be one of "divorced," "married," "single," or "unknown"
    @validator('marital')
    def check_marital(cls, v):
        if v not in {'divorced', 'married', 'single', 'unknown'}:
            raise ValueError("Must be 'divorced', 'married', 'single', or 'unknown', If you want to input 'widowed', type 'divorced' instead")
        return v

    # Input is validated to be either "cellular" or "telephone"
    @validator('contact')
    def check_contact(cls, v):
        if v not in {'cellular', 'telephone'}:
            raise ValueError("Must be 'cellular' or 'telephone'")
        return v
    
    # Input is validated to be either "failure," "nonexistent," or "success"
    @validator('poutcome')
    def check_poutcome(cls, v):
        if v not in {'failure', 'nonexistent', 'success'}:
            raise ValueError("Must be 'failure', 'nonexistent', or 'success'")
        return v

    # The input must be the first 3 characters of the month name
    @validator('month')
    def check_month(cls, v):
        if v not in {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}:
            raise ValueError("Must be 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', or 'dec'")
        return v
    
    # The input must consist of the first 3 characters of the day name and be between Monday and Friday.
    @validator('day_of_week')
    def check_day_of_week(cls, v):
        if v not in {'mon', 'tue', 'wed', 'thu', 'fri'}:
            raise ValueError("Must be 'mon', 'tue', 'wed', 'thu', or 'fri'")
        return v


@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Marketing Classification System - Developed by Elvina"}

@app.post("/predict/")
def predict(data: Data):
    try:
        input_data = pd.DataFrame([data.dict()])

        # Encoding job
        for value in job_train_unique:
            input_data[f"job_{value}"] = 0
        input_data['job_other'] = 0
        if data.job in job_train_unique:
            input_data[f"job_{data.job}"] = 1
        else:
            input_data['job_other'] = 1

        # Encoding marital
        for value in marital_train_unique:
            input_data[f"marital_{value}"] = 0
        if data.marital in marital_train_unique:
            input_data[f"marital_{data.marital}"] = 1

        # Encoding education
        for value in education_train_unique:
            input_data[f"education_{value}"] = 0
        input_data['education_other'] = 0
        if data.education in education_train_unique:
            input_data[f"education_{data.education}"] = 1
        else:
            input_data['education_other'] = 1

        # Encoding month
        for value in month_train_unique:
            input_data[f"month_{value}"] = 0
        if data.month in month_train_unique:
            input_data[f"month_{data.month}"] = 1

        # Encoding day of week
        for value in dow_train_unique:
            input_data[f"day_of_week_{value}"] = 0
        if data.month in month_train_unique:
            input_data[f"day_of_week_{data.day_of_week}"] = 1

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

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)