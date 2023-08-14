import streamlit as st
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np




def preprocess_input(input_data, training_columns):

    # Create a dictionary to map input data's categories to their positions
    category_mapping = {
    'Age_Band': ['1-17', '18-24', '25-44', '45-64', '65-84', '85'],
    'ICD10_Chapter_Code': ['IX','X', 'XI', 'XIV','XVIII','XIX','Other', 'Unknown'],
    'Treatment_Function_Code': ['100', '180','300', '420', 'OTHER', 'Unknown'],
    'AE_HRG': ['Low','Medium', 'High', 'Nothing', 'Unknown']
    # Add other categorical columns and their categories here
    }
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data, columns=training_columns)
    for column in input_data.columns:
        if column in category_mapping.keys():
            for idx, category in enumerate(category_mapping[column]):
                if input_data[column].values[0] == category:
                    input_df[f"{column}_{idx + 1}"] = 1
                else:
                    input_df[f"{column}_{idx + 1}"] = 0

    # Reorder columns to match the training data columns
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # input_df = input_df.fillna(1)
    pd.set_option('display.max_columns', None)
    print(input_df)
    
    return input_df







# Adjust the training_columns list based on the columns used during training
training_columns = ['Age_Band_1', 'Age_Band_2', 'Age_Band_3',
'Age_Band_4', 'Age_Band_5', 'Age_Band_6',
'AE_Num_Investigations',
'AE_Time_Mins',
'AE_Arrival_Mode',
'AE_HRG_1', 'AE_HRG_2', 'AE_HRG_3', 'AE_HRG_4', 'AE_HRG_5',
'ICD10_Chapter_Code_1', 'ICD10_Chapter_Code_2', 'ICD10_Chapter_Code_3',
'ICD10_Chapter_Code_4', 'ICD10_Chapter_Code_5', 'ICD10_Chapter_Code_6',
'ICD10_Chapter_Code_7', 'ICD10_Chapter_Code_8',
'Treatment_Function_Code_1', 'Treatment_Function_Code_2',
'Treatment_Function_Code_3', 'Treatment_Function_Code_4',
'Treatment_Function_Code_5', 'Treatment_Function_Code_6']






# Load the trained model
model = joblib.load('model.joblib')



# Create a function to predict admission
def predict_admission(data):
    data = preprocess_input(data, training_columns)
    # data_scaled = scaler.transform(data)
    prediction = model.predict(data)
    return prediction

# Create the Streamlit app

# Set app title and header
st.title('Patient Admission Prediction :hospital:')
st.write('Enter patient information to predict admission:')

# User input fields
Age_Band = st.selectbox("Choose Age Band", ['1-17', '18-24', '25-44', '45-64', '65-84', '85'])
AE_Num_Investigations = st.slider("Choose No of investigations", 0, 100)
AE_Time_Mins = st.number_input("Choose No of minutes before treatments:", min_value=1, max_value=10000, value=30)
AE_Arrival_Mode = st.selectbox("Choose Arrival Mode", [0, 1, 2])
AE_HRG = st.selectbox("Choose Healthcare Resource Groups", ['Low', 'Medium', 'High', 'Nothing', 'Unknown'])
ICD10_Chapter_Code = st.selectbox("Select ICD10 Chapter Code", ['IX', 'X', 'XI', 'XIV', 'XVIII', 'XIX', 'Other', 'Unknown'])
Treatment_Function_Code = st.selectbox("Choose Treatment Function Code", ['100', '180', '300', '420', 'OTHER', 'Unknown'])


if st.button('Predict'):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'Age_Band': [Age_Band],  'AE_Num_Investigations': [AE_Num_Investigations],
    'AE_Time_Mins': [AE_Time_Mins], 'AE_Arrival_Mode': [AE_Arrival_Mode], 'AE_HRG': [AE_HRG],
    'ICD10_Chapter_Code': [ICD10_Chapter_Code],
    'Treatment_Function_Code': [Treatment_Function_Code]
    })
    # Add more columns to the DataFrame based on the input fields
    
    # Make the prediction
    prediction = predict_admission(input_data)
    
    if prediction[0] == 0:
        st.write('Prediction: Not Admitted')
    else:
        st.write('Prediction: Admitted')


