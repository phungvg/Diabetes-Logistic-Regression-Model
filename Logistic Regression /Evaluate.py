import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib

"""Confusion matrix -- TP, TN, FP, FN"""
def confusion_matrix(cm,title):
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt ='d',cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig ('Confusion_Matrix.png')
    plt.show()

"""Load the model"""
def evaluate_model():
    test_df = pd.read_csv('/Users/panda/Documents/APM/RiskScorePrediction /Data/Test/Test_data.csv')
    pass

    #Calculate metrics -- accuracy, f1 score, confusion matrix

