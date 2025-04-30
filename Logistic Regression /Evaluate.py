import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

##Visualization: Saneky Diamgram, Confusion Matrix, Box Plot , Feature importance bar chart 

# Evaluate the model on the held-out test set
def evaluate_model():
    train_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Train/Train_data.csv'
    val_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Val/Val_data.csv'
    
    model = train_model(train_dir,val_dir)
    test_path = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Test/Test_data.csv'
    y_pred = model.predict(x_test)

    print('Test accuracy:', accuracy_score(y_test, y_pred))
    print('Test F1 score:', f1_score(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.show()

if __name__ == '__main__':
    evaluate_model()