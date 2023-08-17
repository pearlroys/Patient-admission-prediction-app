import streamlit as st
import streamlit as st
import joblib
import pandas as pd
import numpy as np





def preprocess_input(input_data, training_columns):

    # Create a dictionary to map input data's categories to their positions
    category_mapping = {
    'Age_Band': ['1-17', '18-24', '25-44', '45-64', '65-84', '85+'],
    'AE_HRG': ['High', 'Low','Medium', 'Nothing', 'Unknown'],
    'ICD10_Chapter_Code': ['IX', 'Other', 'Unknown', 'X', 'XI', 'XIV', 'XIX','XVIII'],
    'Treatment_Function_Code': ['100', '180','300', '420', 'OTHER', 'Unknown']
    
    # Add other categorical columns and their categories here
    }
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data, columns=training_columns)
    for column in input_data.columns:
        if column in category_mapping.keys():
            for idx, category in enumerate(category_mapping[column]):
                if input_data[column].values[0] == category:
                    input_df[f"{column}_{category}"] = 1
                else:
                    input_df[f"{column}_{category}"] = 0

    # Reorder columns to match the training data columns
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # input_df = input_df.fillna(1)
    pd.set_option('display.max_columns', None)
    print(input_df)
    
    return input_df


# Adjust the training_columns list based on the columns used during training
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
AE_Time_Mins = st.number_input("Choose No of minutes before treatments:", min_value=1, max_value=10000, value=30)
AE_Num_Investigations = st.slider("Choose No of investigations", 0, 100)
AE_Arrival_Mode = st.selectbox("Choose Arrival Mode", [0, 1, 2])
Age_Band = st.selectbox("Choose Age Band", ['1-17', '18-24', '25-44', '45-64', '65-84', '85'])
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
        st.write('Prediction: Admission not required')
    else:
        st.write('Prediction: Requires Admission')


