import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("animal_knn_model.joblib")

st.set_page_config(page_title="Animal Species Predictor", layout="centered")

st.title("🐾 Animal Species Predictor")
st.write("Enter animal characteristics to predict its species.")

# Input fields
body_weight = st.number_input("Body Weight (kg)", min_value=0.0, value=10.0)
body_length = st.number_input("Body Length (cm)", min_value=0.0, value=50.0)
leg_count = st.number_input("Leg Count", min_value=0, value=4, step=1)

has_fur = st.selectbox("Has Fur?", [0, 1])
has_feathers = st.selectbox("Has Feathers?", [0, 1])
can_fly = st.selectbox("Can Fly?", [0, 1])
lays_eggs = st.selectbox("Lays Eggs?", [0, 1])

lifespan = st.number_input("Average Lifespan (years)", min_value=0.0, value=15.0)

# Prediction button
if st.button("Predict Species"):
    input_data = np.array([[
        body_weight,
        body_length,
        leg_count,
        has_fur,
        has_feathers,
        can_fly,
        lays_eggs,
        lifespan
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Species: **{prediction}**")
