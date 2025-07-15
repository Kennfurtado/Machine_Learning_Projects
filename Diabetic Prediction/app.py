import streamlit as st
import pandas as pd
import pickle
import numpy as np
#Load the trained data from a file
df = pd.read_csv('diabetes.csv')
model = pickle.load(open('diabetes_model.pkl', 'rb'))
st.title('Diabetes Prediction using Decision tree')

st.sidebar.header('Enter Patient info:')

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=1)
Glucose = st.sidebar.number_input("Glucose", min_value=0)
BloodPressure = st.sidebar.number_input("BloodPressure", min_value=0)
SkinThickness = st.sidebar.number_input("SkinThickness", min_value=0)
Insulin = st.sidebar.number_input("Insulin", min_value=0)
BMI = st.sidebar.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0)
Age = st.sidebar.number_input("Age", min_value=1)

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction Result: Diabetic")
    else:
        st.success("Prediction Result: Not Diabetic")
