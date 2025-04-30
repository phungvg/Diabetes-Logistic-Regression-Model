#Total: 100001 cases
# Training set shape:  (65000, 8) 65%
# Validation set shape:  (9999, 8) 10%
# Test set shape:  (25001, 8) 25%

import pandas as pd
import os
from sklearn.model_selection import train_test_split
# Load raw data and split
gt = '/Users/panda/Documents/APM/RiskScorePrediction /Data/diabetes_prediction_dataset.csv'
out_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data'


def split_data(df):
    # Separate features (X) and target (y)
    x = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # First split: 65% train, 35% remaining
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y,
        train_size=0.65,
        random_state=42,
        stratify=y  # keep class distribution
    )

    # Second split: 28.57% of remaining (~10% overall) as validation, rest as test
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test,
        train_size=0.2857,
        random_state=42,
        stratify=y_val_test
    )

    # Save paths
    train_path = os.path.join(out_dir, 'Train', 'Train_data.csv')
    val_path = os.path.join(out_dir, 'Val', 'Val_data.csv')
    test_path = os.path.join(out_dir, 'Test', 'Test_data.csv')

    # Ensure directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # Save splits to CSV files
    pd.concat([x_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([x_val, y_val], axis=1).to_csv(val_path, index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(test_path, index=False)

    print('Training set shape:     ', x_train.shape)
    print('Validation set shape:   ', x_val.shape)
    print('Test set shape:         ', x_test.shape)

    # return train_path, val_path, test_path

# Run splitting when this file is executed directly
if __name__ == '__main__':
    df = pd.read_csv(gt)
    split_data(df)