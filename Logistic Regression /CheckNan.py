import pandas as pd

for file in ['Train/Train_data.csv', 'Val/Val_data.csv', 'Test/Test_data.csv']:
    path = f'/Users/panda/Documents/APM/RiskScorePrediction /Data/{file}'
    df = pd.read_csv(path)
    print(f"\nChecking {file}:")
    print(df.isna().sum())