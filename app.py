import streamlit as st
import pickle  # or joblib

# Load your model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input
user_input = st.number_input("Enter a value")

# Prediction
prediction = model.predict([[user_input]])
st.write("Prediction:", prediction[0])
