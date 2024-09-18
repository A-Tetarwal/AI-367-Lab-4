import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from google.colab import files

uploaded = files.upload()

zip_file_path = '/content/thyroid+disease.zip'
extracted_folder_path = '/content/thyroid_disease/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

extracted_files = os.listdir(extracted_folder_path)
print(f"Extracted files: {extracted_files}")

data_path = os.path.join(extracted_folder_path, 'new-thyroid.data')

column_names = ['Class', 'T3', 'T4', 'TSH']
data = pd.read_csv(data_path, names=column_names, delim_whitespace=True)

class_map = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
data['Class'] = data['Class'].map(class_map)

scaler = StandardScaler()
data[['T3', 'T4', 'TSH']] = scaler.fit_transform(data[['T3', 'T4', 'TSH']])

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['Class'], test_size=0.3, random_state=42)
X_train['Class'] = y_train

hc = HillClimbSearch(X_train)
best_model = hc.estimate(scoring_method=BicScore(X_train))

model = BayesianNetwork(best_model.edges())

required_nodes = ['T3', 'T4', 'TSH', 'Class']
for node in required_nodes:
    if node not in model.nodes():
        model.add_node(node)

model.fit(X_train, estimator=MaximumLikelihoodEstimator, state_names={'Class': ['Hyperthyroid', 'Hypothyroid', 'Normal']})

inference = VariableElimination(model)

def predict(model, inference, X_test):
    y_pred = []
    for _, sample in X_test.iterrows():
        evidence = sample.to_dict()
        query = inference.map_query(variables=['Class'], evidence=evidence)
        y_pred.append(query['Class'])
    return y_pred

y_pred = predict(model, inference, X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')