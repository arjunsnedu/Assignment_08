import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.write("Current directory:", os.getcwd())
st.write("Files available:", os.listdir())

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page description and titles
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding Inputs
sex_male = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([[
    pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
]], columns=[
    "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male",
    "Embarked_Q", "Embarked_S"
])

# Prediction
if st.button("Predict Survival Probability"):
    input_scaled = scaler.transform(input_data)
    survival_prob = model.predict_proba(input_scaled)[0][1]

    st.success(f"🧍 Survival Probability: **{survival_prob:.2%}**")

    if survival_prob >= 0.5:
        st.write("🟢 Prediction: **Likely to Survive**")
    else:
        st.write("🔴 Prediction: **Unlikely to Survive**")