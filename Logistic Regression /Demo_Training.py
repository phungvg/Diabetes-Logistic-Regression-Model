"""Workflow
1. Load dataset from Kaggle(.csv files)
2. Preprocess(handle missing values, many duplicate info, scale numerical data,mea,median, remove data if needed)
3. Split into training, var,test sets (70,15,15 -- range can be fixed)
4. Training model
5. Inference (set up, loading, get data, target,post process), goal is to predict risk score for testing
6. Evaluate model peformance, plot the result with actual vs predicted
"""
##Demo Training for Diabetes Training Model 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import time
import warnings
import os
import warnings

# ----------------------------------------------------------------------------
"""Load datadset"""
# ----------------------------------------------------------------------------
gt ='/Users/panda/Documents/APM/RiskScorePrediction /Dataset/diabetes_prediction_dataset.csv'
df = pd.read_csv(gt)

# # #Checking for Null values
# print("Missing values in the table:\n")
# print(df.isna().sum())

## Level of impact of diabetes are
# Blood Glucose Level (most important), above 100 - 125 mg/dL is prediabetes, 126 is diabetes, less than 100 is normal.
# HbA1c Level – Reflects long-term glucose control, crucial for diagnosis, in %, if above 6.5 is diabetes, below is good
#Highest is 9, mean is 5.5. between 5.7-6.4 is considered pre diabetes. Normal is below 5.7
# BMI – Higher BMI is strongly associated with Type 2 diabetes.
# Hypertension – Common in diabetic patients due to metabolic issues.
# Heart Disease – Often coexists with diabetes; both share risk factors.
# Age – Older individuals are more likely to develop diabetes.
# Smoking History – Smoking increases insulin resistance, indirectly contributing to diabetes.
# Gender – May have a minor effect, but generally less predictive than the others.

#Check calculation involving
# print(df.describe())
# max_column = df['HbA1c_level'].max()
# min_column = df['HbA1c_level'].min()
# print(f'Max value of HbA1c Level is {max_column} and min value is {min_column}')
# print(' ')

# ----------------------------------------------------------------------------
"""Preprocessing"""
# ----------------------------------------------------------------------------
def preprocess(df):
    #Drop rows with missing values
    # df = df.dropna() 

    #Drop duplicate
    # df = df.drop_duplicates()
    # print(f"Removed {df.duplicated().sum()} duplicate rows")

    """Encoding -- gender, and smoking history since they are numerical data"""
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])

    """Scale numerical features"""
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    # df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df,scaler


# ----------------------------------------------------------------------------
"""Building model"""
# ----------------------------------------------------------------------------
def train_model(train_dir,val_dir,output_dir='.'):
    #Seperate data into features and target
    #Features
    feature_col =[ 'hypertension', 'smoking_history',
        'bmi', 'HbA1c_level', 'blood_glucose_level'
    ]
    train_df = pd.read_csv(train_dir)
    val_df = pd.read_csv(val_dir)

    X_train = train_df[feature_col]
    Y_train = train_df['diabetes']

    #Target
    y_train = val_df[feature_col]
    y_val = val_df['diabetes']

    #Preprocessing features
    X_train_processed, scaler = preprocess (X_train, feature_col)
    X_val_processed, _ = preprocess (y_train, feature_col)
def learning_curve(train)

# Main script
# if __name__ == '__main__':
#     # ----------------------------------------------------------------------------
#     # Set up data
#     # ----------------------------------------------------------------------------
#     start_time = time.time()
#     start_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
#     print('-------------------------------------')  
#     print('Running main script at', start_time_str, '\n')
    
#     # gt ='/Users/panda/Documents/APM/RiskScorePrediction /Dataset/diabetes_prediction_dataset.csv'
#     # df = pd.read_csv(gt)

#     # # #Display info of the tbl
#     print("All the categories in the table \n")
#     print(df.info())
#     print('-------------------------------------')  


#    train_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Train'
#    val_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Val'
#    output_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Output'

#    train_model(train_dir,val_dir,output_dir)