import joblib
from bip import preprocess_input

# Load the trained model
model = joblib.load('model.joblib')

training_columns = ['AE_Time_Mins',
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


# Create a function to predict admission
def predict_admission(data):
    data = preprocess_input(data, training_columns)
    # data_scaled = scaler.transform(data)
    prediction = model.predict(data)
    return prediction
