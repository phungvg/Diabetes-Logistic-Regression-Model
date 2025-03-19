
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
#Total: 100001 cases
# Training set shape: (62494, 8) (65.0%)
# Validation set shape: (9614, 8) (10.0%)
# Test set shape: (24038, 8) (25.0%)

def split_data(df):
    #Toal 100001 cases, split into train, test, val with 65,10,25 ratio 
    #temp 9
    X = df.drop('diabetes', axis =1) #features, axis =1 means drop column, if axis = 0 means drop row
    Y = df['diabetes'] #target

    # Split off 65% for training, leaving 35% for val + test
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, train_size=0.65, random_state=42)
    
    # Split the remaining 35% into 10% val and 25% test
    # 10% of total is 0.10 / 0.35 = 0.2857 of the val_test set
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, train_size=0.2857, random_state=42)

    #Path address for saving data after split 
    train_path = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Train'
    val_path = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Val'
    test_path = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Test'

    #Combine features and target into df
    train_df = pd.concat([X_train, Y_train], axis =1)
    val_df = pd.concat([X_val, Y_val], axis =1)
    test_df = pd.concat([X_test, Y_test], axis =1)

    #Save as CSV files
    train_df.to_csv(os.path.join(train_path, 'Train_data.csv'), index = False) #Index = False means no index, otherwise index will be saved as well
    val_df.to_csv(os.path.join(val_path, 'Val_data.csv'), index = False)
    test_df.to_csv(os.path.join(test_path, 'Test_data.csv'), index = False)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if __name__ == "__main__":
    df = pd.read_csv("/Users/panda/Documents/APM/RiskScorePrediction /Data/diabetes_prediction_dataset.csv")
    split_data(df)