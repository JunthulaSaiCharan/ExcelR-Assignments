import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open('log_model.pkl', 'rb'))

st.title("Titanic Survival Prediction")

# User inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare Paid', 0.0, 600.0, 30.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encode inputs
sex = 1 if sex == 'male' else 0
embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Create input array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button('Predict'):
    result = model.predict(features)[0]
    st.success('Survived' if result == 1 else 'Did Not Survive')
