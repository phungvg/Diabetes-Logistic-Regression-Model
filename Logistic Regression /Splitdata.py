#Total: 100001 cases
# Training set shape:  (65000, 8) 65%
# Validation set shape:  (9999, 8) 10%
# Test set shape:  (25001, 8) 25%

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop('diabetes', axis=1)
    Y = df['diabetes']
    
    # Split into 65% train, 35% remaining
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.65, random_state=42, stratify=Y)
    # Split remaining into 10% val (0.2857 of 35%) and 25% test
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.2857, random_state=42, stratify=Y_val_test)

    # Save paths
    out_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data'
    train_path = os.path.join(out_dir, 'Train', 'Train_data.csv')
    val_path = os.path.join(out_dir, 'Val', 'Val_data.csv')
    test_path = os.path.join(out_dir, 'Test', 'Test_data.csv')

    # Ensure directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # Combine and save
    pd.concat([X_train, Y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_val, Y_val], axis=1).to_csv(val_path, index=False)
    pd.concat([X_test, Y_test], axis=1).to_csv(test_path, index=False)

    print("Training set shape: ", X_train.shape)
    print("Validation set shape: ", X_val.shape)
    print("Test set shape: ", X_test.shape)

# Load raw data and split
gt = '/Users/panda/Documents/APM/RiskScorePrediction /Data/diabetes_prediction_dataset.csv'
df = pd.read_csv(gt)
print("Raw data missing values:\n", df.isna().sum())
split_data(df)
