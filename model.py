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

df = pd.read_csv('cleansss_data.csv')
print(df.columns)




rf_features = ['AE_Time_Mins',
'AE_Num_Investigations','AE_Arrival_Mode',
'Age_Band_1-17', 'Age_Band_18-24',
'Age_Band_25-44', 'Age_Band_45-64', 'Age_Band_65-84', 'Age_Band_85+',
'AE_HRG_High', 'AE_HRG_Low', 'AE_HRG_Medium', 'AE_HRG_Nothing',
'AE_HRG_Unknown', 'ICD10_Chapter_Code_IX', 'ICD10_Chapter_Code_Other',
'ICD10_Chapter_Code_Unknown', 'ICD10_Chapter_Code_X',
'ICD10_Chapter_Code_XI', 'ICD10_Chapter_Code_XIV',
'ICD10_Chapter_Code_XIX', 'ICD10_Chapter_Code_XVIII',
'Treatment_Function_Code_100', 'Treatment_Function_Code_180',
'Treatment_Function_Code_300', 'Treatment_Function_Code_420',
'Treatment_Function_Code_OTHER', 'Treatment_Function_Code_Unknown']

X = df[rf_features]
print(X.isnull().sum())
y = df['Admitted_Flag']
# X = df.drop('Admitted_Flag', axis=1)
# y = df['Admitted_Flag']
# print(X.columns)
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