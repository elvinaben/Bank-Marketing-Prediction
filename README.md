# Bank Marketing Prediction Model

This project develops a predictive model to assess customer potential for bank offers based on various attributes such as age, job, marital status, education level, credit default status, housing loan, personal loan, contact method, last contact duration, and previous campaign outcomes.

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the data distribution and identify issues such as class imbalance and missing values. The age variable showed the highest frequency in the range of 30-40 years, and class imbalance was evident with more 'no' than 'yes' responses for the target variable.

## Data Preprocessing
- **Train-Test Split**: 80:20 ratio.
- **Missing Values**: Imputed using mode for the job variable and median for the duration variable (due to outliers).
- **Categorical Encoding**: Applied one-hot encoding for categorical features and label encoding for binary features.
- **Standardization**: Standard Scaler was used for numeric columns to ensure consistent units.
- **Class Imbalance**: SMOTE was applied on the training set to generate synthetic samples for the minority class, addressing class imbalance.

## Model Selection
Two models were developed:
1. **Decision Tree**: Hyperparameters were tuned using GridSearchCV.
2. **Bagging Decision Tree**: Used the decision tree as the base estimator. The bagging method aims to improve the overall performance by reducing variance through ensemble learning.

<img width="935" alt="image" src="https://github.com/user-attachments/assets/c4725386-1682-471f-b2ae-b7faec0c5411">

Due to the class imbalance (more "no" responses than "yes"), accuracy was not the main evaluation metric, as it could be misleading. The bank's goal is to identify customers likely to subscribe to long-term deposits, so it’s important to prioritize:

Precision, to ensure we are targeting customers truly likely to subscribe, avoiding wasted resources.
Recall, to capture most potential customers and not miss out on opportunities.
Both false positives and false negatives are critical. Therefore, I used the F1 score as the main metric, balancing precision and recall.

The decision tree achieved an F1-Score of 50% for the "yes" class, while the bagging model improved to 53%, outperforming in both precision and recall. Additionally, the bagging model consistently surpassed the decision tree across all metrics, leading to its selection for deployment in the machine learning API. This model helps predict customer potential for bank offers, optimizing marketing efforts.

## Deployment
The **bagging decision tree model** was selected for deployment in **FastAPI** and **Streamlit**, aimed at predicting whether a customer is a potential target for the bank's long-term deposit offers.

### FastAPI Deployment
#### Sample of a potential customer
<img width="887" alt="image" src="https://github.com/user-attachments/assets/aec041b4-43a4-4be0-a638-d383b8173ddb">

**Input :** 

<img width="354" alt="image" src="https://github.com/user-attachments/assets/4f5e77a5-4e0b-446a-9c87-ee0635513d89"> 

**Output :** 

<img width="511" alt="image" src="https://github.com/user-attachments/assets/4597d4ef-bbdb-4744-b72f-c91a04af3eff">

#### Sample of a non-potential customer
<img width="843" alt="image" src="https://github.com/user-attachments/assets/49efa1b8-cfa0-4ac0-95b4-b96153d50dc4">

**Input :** 

<img width="205" alt="image" src="https://github.com/user-attachments/assets/ee023d12-6da0-4b3c-bc6f-2a82b04db52e"> 

**Output :** 

<img width="310" alt="image" src="https://github.com/user-attachments/assets/9f407993-5201-489a-8569-81972edb5050">

### Streamlit Deployment
<img width="372" alt="image" src="https://github.com/user-attachments/assets/4fe2199c-c64e-4436-8533-8cd4fae2c4a4">

<img width="370" alt="image" src="https://github.com/user-attachments/assets/d61331eb-f29d-4f64-b3cb-2b62feb18a4b"> 

<img width="364" alt="image" src="https://github.com/user-attachments/assets/133bd7e2-88e1-4c99-8992-d2bff921d1c5">





