import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 90)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=0.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 0, 4)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [int(is_active_member)],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Combine all features
input_data = pd.concat([input_data, geo_encoded_df], axis=1)


input_data = input_data[scaler.feature_names_in_]


input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f"### Prediction Probability of Churn: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.error("The customer is likely to churn")
else:
    st.success("The customer is not likely to churn")