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
from sklearn.linear_model import LogisticRegression 
import time
import warnings
import os
import warnings

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


# ----------------------------------------------------------------------------
"""Preprocessing"""
# ----------------------------------------------------------------------------
def preprocess(df):
    #Drop rows with missing values
    df = df.dropna() 

    #Drop duplicate
    df = df.drop_duplicates()
    print(f"Removed {df.duplicated().sum()} duplicate rows")

    """Encoding -- gender, and smoking history since they are numerical data"""
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])
    
    #Age bins 
    bins = [0, 20,40,60, np.inf]
    labels = ['0-20', '21-40', '41-60', '61+']
    df['age_groups'] = pd.cut(df['age'], bins=bins, labels=labels)
    df = df.drop('age', axis = 1)

    #Numerical features
    numerical_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_cols = ['hypertension', 'heart_disease', 'smoking_history', 'gender', 'age_group']

    #Check for NaNs
    print("Missing values before imputation:\n", df.isna().sum())
    
    #Handle for Nans

    return df

# ----------------------------------------------------------------------------
"""Learning rate"""
# ----------------------------------------------------------------------------
# def learning_rate(train_sz, train_scores, val_scores,output_dir ='.' ):
#     "Observe for each time that the model generate"
#     print("Train size: ", train_sz)
#     print("Train scores: ", train_scores)
#     print("Validation scores: ", val_scores)
#     print(" ")

#     plt.figure(figsize=(12,6))
#     plt.plot(train_sz, train_scores, label='Training Score', marker='o')
#     plt.plot(train_sz, val_scores, label='Validation Score', marker='-')
#     plt.title('Learning Curve for Diabetes Prediction Over Time')
#     plt.xlabel('Training')
#     plt.ylabel('Accuracy Score')
#     plt.legend(loc = 'best') #loc is for legend, best is for best position 
#     plt.grid()
#     plt.savefig(output_dir + '/learning_curve.png'))
#     plt.show()

# ----------------------------------------------------------------------------
"""Building model"""
# ----------------------------------------------------------------------------
def train_model(train_dir, val_dir, test_dir, output_dir='.'):
    #Load data 
    train_df = pd.read_csv(train_dir)
    val_df = pd.read_csv(val_dir)
    test_df = pd.read_csv(test_dir)

    #Features are all comlumns impact to the target 
    feature_cols = ['hypertension', 'heart_disease', 'smoking_history', 
                'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'age_group']

    #Preprocess data with imputation
 


# Main script
if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    # Set up data
    # ----------------------------------------------------------------------------
    start_time = time.time()
    start_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(start_time))
    print('-------------------------------------')  
    print('Running main script at', start_time_str, '\n')
    train_dir = '/Users/panda/Documents/APM/Testing/Train/Train_data.csv'
    gt ='/Users/panda/Documents/APM/RiskScorePrediction /Data/diabetes_prediction_dataset.csv'
    # df = pd.read_csv(gt)
    df = pd.read_csv(train_dir)
    # df = pd.read_csv(val_dir)

    # # #Display info of the tbl
    print("All the categories in the table \n")
    print(df.info())
    print('-------------------------------------')  
    print(" ")
    #Display missing values or having Nans
    print("Missing values in the table \n", df.isnull().sum())
    print('-------------------------------------') 


    train_dir = '/Users/panda/Documents/APM/Testing/Train/Train_data.csv'
    val_dir = '/Users/panda/Documents/APM/Testing/Val/Val_data.csv'
    output_dir = '/Users/panda/Documents/APM/RiskScorePrediction/Output'

#    train_model(train_dir,val_dir,output_dir)