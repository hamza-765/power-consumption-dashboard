import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Energy Predictor", layout="wide")
st.title("⚡ Energy Prediction")

model = joblib.load('energy_model.pkl')
df = pd.read_csv('newenergydata.csv')

st.write("Data preview:")
st.dataframe(df.head())

features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
           'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
           'hour', 'day', 'month', 'weekday', 'is_weekend', 'total_sub_metering']

st.write("Enter values:")
input_data = {}

for col in features:
    avg = float(df[col].mean())
    input_data[col] = st.number_input(col, value=avg)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted: {prediction:.2f} kWh")

# If you want to show the actual Global_active_power average
if 'Global_active_power' in df.columns:
    st.write(f"Average Global_active_power: {df['Global_active_power'].mean():.2f} kWh")