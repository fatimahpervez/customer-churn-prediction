import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('Customer Churn Predictor')

tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0)

contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

if st.button('Predict Churn'):
    input_dict = {col: 0 for col in model_columns}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges

    if contract == 'One year':
        input_dict['Contract_One year'] = 1
    elif contract == 'Two year':
        input_dict['Contract_Two year'] = 1

    if internet_service == 'Fiber optic':
        input_dict['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        input_dict['InternetService_No'] = 1

    if online_security == 'Yes':
        input_dict['OnlineSecurity_Yes'] = 1
    elif online_security == 'No internet service':
        input_dict['OnlineSecurity_No internet service'] = 1

    if tech_support == 'Yes':
        input_dict['TechSupport_Yes'] = 1
    elif tech_support == 'No internet service':
        input_dict['TechSupport_No internet service'] = 1

    if payment_method == 'Electronic check':
        input_dict['PaymentMethod_Electronic check'] = 1
    elif payment_method == 'Mailed check':
        input_dict['PaymentMethod_Mailed check'] = 1
    elif payment_method == 'Credit card (automatic)':
        input_dict['PaymentMethod_Credit card (automatic)'] = 1

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f'This customer is likely to churn. (Probability: {probability:.1%})')
    else:
        st.success(f'This customer is likely to stay. (Probability of churn: {probability:.1%})')

