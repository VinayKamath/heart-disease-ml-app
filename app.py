# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:49:49 2026

@author: vinay
"""
import joblib
import streamlit as st 
import pandas as pd

classifier = joblib.load("final_model.joblib")


def predict_heart_disease(age, trestbps, chol, thalch, oldpeak, sex, cp, fbs, restecg, exang, slope):
    input_data = pd.DataFrame({
        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol],
        "thalch": [thalch],
        "oldpeak": [oldpeak],
        "sex": [sex],
        "cp": [cp],
        "fbs": [fbs],
        "restecg": [restecg],
        "exang": [exang],
        "slope": [slope]
    })
    prediction = classifier.predict(input_data)
    print(prediction)
    return prediction
    

def main():
    st.title("UCI Heart Disease Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">UCI Heart Disease Detection ML App</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.text_input("Age")
    trestbps = st.text_input("trestbps")
    chol = st.text_input("Cholestrol Lvl (chol)")
    thalch = st.text_input("Max Heart Rate (thalch)")
    oldpeak = st.text_input("Oldpeak")
    sex = st.text_input("Sex")
    cp = st.text_input("Chest Pain Type (cp)")
    fbs = st.text_input("Fasting Blood Sugar (fbs)")
    restecg = st.text_input("Resting ECG result")
    exang = st.text_input("Exercise Induced Angina (exang)")
    slope = st.text_input("Slope of Peak Exercise ST segment (slope)")
    
    result = ""
    if st.button("Predict"):
        result = predict_heart_disease(age, trestbps, chol, thalch, oldpeak, sex, cp, fbs, restecg, exang, slope)
    
    st.success("The output is {}".format(result))
    
    
    
    
    
if __name__ == '__main__':
    main()



