import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

zip_file_path = r'C:\Users\anujs\Downloads\thyroid+disease.zip'
extracted_folder_path = r'C:\Users\anujs\Downloads\thyroid\disease'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

extracted_files = os.listdir(extracted_folder_path)
data_path = os.path.join(extracted_folder_path, 'new-thyroid.data')

column_names = ['Class', 'T3', 'T4', 'TSH']
data = pd.read_csv(data_path, names=column_names, delim_whitespace=True)

class_map = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
data['Class'] = data['Class'].map(class_map)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['Class'], test_size=0.3, random_state=42)
X_train['Class'] = y_train

model = BayesianNetwork([('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')])
model.fit(X_train, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

def predict(model, X_test):
    y_pred = []
    for _, sample in X_test.iterrows():
        query = inference.map_query(variables=['Class'], evidence=sample.to_dict())
        y_pred.append(query['Class'])
    return y_pred

y_pred = predict(inference, X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy