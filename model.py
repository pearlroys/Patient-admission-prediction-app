import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
import joblib
import json
import os
import category_encoders as ce

df = pd.read_csv('cleanss_data.csv')
# df.info()



rf_features = [
'Age_Band_1', 'Age_Band_2', 'Age_Band_3',
'Age_Band_4', 'Age_Band_5', 'Age_Band_6',
'AE_Num_Investigations',
'AE_Time_Mins',
'AE_Arrival_Mode',
'AE_HRG_1', 'AE_HRG_2', 'AE_HRG_3', 'AE_HRG_4', 'AE_HRG_5',
'ICD10_Chapter_Code_1', 'ICD10_Chapter_Code_2', 'ICD10_Chapter_Code_3',
'ICD10_Chapter_Code_4', 'ICD10_Chapter_Code_5', 'ICD10_Chapter_Code_6',
'ICD10_Chapter_Code_7', 'ICD10_Chapter_Code_8','Treatment_Function_Code_1', 'Treatment_Function_Code_2',
'Treatment_Function_Code_3', 'Treatment_Function_Code_4',
'Treatment_Function_Code_5', 'Treatment_Function_Code_6']


X = df[rf_features]
print(X.isnull().sum())
y = df['Admitted_Flag']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model1 = RandomForestClassifier(random_state=42, max_depth = 1, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 100)
rf_model1.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = rf_model1.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy4 = accuracy_score(y_test, y_pred)
auc_score4 = roc_auc_score(y_test, y_pred)
f14 = f1_score(y_test, y_pred)
precision4 = precision_score(y_test, y_pred)
recall4 = recall_score(y_test, y_pred)

# Print the metrics
print("F1 Score:", f14)
print("Precision:", precision4)
print("Recall:", recall4)
print("Accuracy:", accuracy4)
print(f"AUC score: {auc_score4}")
print(y_pred)
# Save the trained model
model_path = os.path.join('/Users/pearl/Downloads/bips', "model.joblib")
joblib.dump(rf_model1, model_path)