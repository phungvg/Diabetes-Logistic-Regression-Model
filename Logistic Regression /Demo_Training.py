"""Workflow
Dataset from Kaggle(.csv files)
Split into training, var,test sets (70,15,15 -- range can be fixed)

"""
##Demo Training for Diabetes Training Model 

import time
import pandas as pd

from sklearn.pipeline import Pipeline # To chain preprocessing and modeling steps
from sklearn.compose import ColumnTransformer # ColumnTransformer is for handling different types of data
from sklearn.impute import SimpleImputer # SimpleImputer is for handling missing values
# OneHotEncoder is for categorical data, StandardScaler is for numerical data to be scaled
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV # GridSearchCV is for hyperparameter tuning
from sklearn.metrics import accuracy_score, classification_report 



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
    df['gender'] = le.fit_transform(df['gender']) #male/female -> 0/1
    df['smoking_history'] = le.fit_transform(df['smoking_history']) # categories → numeric codes
    
    #Age bins 
    bins = [0, 20,40,60, 80] #Age range
    labels = ['0-20', '21-40', '41-60', '61+']
    df['age_groups'] = pd.cut(df['age'], bins=bins, labels=labels)
    df = df.drop('age', axis = 1)

    #Check for NaNs
    print("Missing values before imputation:\n", df.isna().sum())
    
    return df


# ----------------------------------------------------------------------------
"""Load data"""
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path) 
    df = preprocess(df)
    x = df.drop('diabetes', axis=1) #drop diabetes column 
    y = df['diabetes'] #take the rest except this target column
    return x, y

# ----------------------------------------------------------------------------
"""Build preprocessor -- handling missing values"""
# ----------------------------------------------------------------------------
def build_preprocessor():
    numeric_features = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['hypertension', 'heart_disease', 'smoking_history', 'gender', 'age_group']

    numeric_transformer = Pipeline([
        ('imputer',SimpleImputer(strategy='median')), #fill NaNs with median
        ('scaler',StandardScaler()), #afternimputation, each col is centered to mean 0 and scaled to unit variance
        ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), #fill in with mode(appears most often)
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])
    return preprocessor


# ----------------------------------------------------------------------------
"""Building model"""
# ----------------------------------------------------------------------------
def train_model(train_dir, val_dir):
    #Load data 
    x_train, y_train = load_data(train_dir)
    x_val, y_val = load_data(val_dir)

    ## Build pipeline: preprocessing + lr
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(solver = 'liblinear', max_iter=10000)),
         ## solver = 'liblinear' for sparse matrices, or just means faster training
        ])

    ##Hyperparameter grid for regularization strength & type
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10], 
        ## c = 1/λ
        #Picking this range: smaller C->larger λ, max 10 is to fit more flexibly (lower bias but potentially higher variance)
        #regularization strength, means inverse of regularization strength
        #Smaller -> Stronger regulirazaation, best trade-off bts under --over fitting
        'clf__penalty': ['l1', 'l2'], #L1 -> L2
    } 
    ##minw[LogLoss(w)+λ∥w∥pp] --> p=1 gives L1 , p=2 gives L2, λ controls how strong the penalty is -> stiffer penalty -> smaller w
    ## L1 is Lasso -- Penalizes weights linearly, tends to shirnk all weights toweard 0, but rarely make them exactly 0
    ## L2 is Ridge -- Penalizes large weights quadratically, many weights turns exactly 0
    
    
    
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

    # # #Display info of the tbl
    print('-------------------------------------')  
    print(" ")
    #Display missing values or having Nans
    # print("Missing values in the table \n", df.isnull().sum())


    train_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Train/Train_data.csv'
    val_dir = '/Users/panda/Documents/APM/RiskScorePrediction /Data/Val/Val_data.csv'
    # output_dir = '/Users/panda/Documents/APM/RiskScorePrediction/Output'

    model = train_model(train_dir, val_dir)
    elapsed = time.time() - start_time
    print(' ')
    print(f'Training completed in {elapsed:.1f}s')