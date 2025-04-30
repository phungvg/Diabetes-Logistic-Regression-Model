# --- Inference.py ---
import sys
import pandas as pd
from Demo_Training import preprocess, train_model

# Predict new data
def inference(df_new):
    x_new,_=preprocess(df_new)
    train_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Train/Train_data.csv'
    val_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Val/Val_data.csv'
    model = train_model(train_dir,val_dir)
    preds=model.predict(x_new)
    probs=model.predict_proba(x_new)[:,1]
    return preds,probs

if __name__=='__main__':
    df=pd.read_csv('/Users/panda/Documents/APM/RiskScorePrediction /Data/Test/Test_data.csv')
    p,pr=inference(df)
    print('Predictions:',p)
    print('Probabilities:',pr)