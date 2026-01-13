import streamlit as st
import numpy as np
import joblib

# Load models
lr_model = joblib.load("linear_regression_charging_time.pkl")
rf_model = joblib.load("random_forest_trip_feasibility.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="EV Bus Battery ML Platform", layout="centered")

st.title("🚌 EV Bus Battery Health & Trip Feasibility System")
st.write("EV-Bus Intelligent Fleet Analytics ")

st.header("🔋 Battery Input Parameters")

# User Inputs
SOC = st.slider("State of Charge (SOC %)", 0, 100, 50)
SOH = st.slider("State of Health (SOH %)", 0, 100, 80)
terminal_voltage = st.number_input("Terminal Voltage (V)", value=400.0)
battery_current = st.number_input("Battery Current (A)", value=120.0)
battery_temp = st.slider("Battery Temperature (°C)", -10, 80, 30)
ambient_temp = st.slider("Ambient Temperature (°C)", -10, 60, 25)
internal_resistance = st.number_input("Internal Resistance (Ohm)", value=0.02)
charging_efficiency = st.slider("Charging Efficiency (%)", 0, 100, 90)
cycle_degradation = st.slider("Cycle Degradation (%)", 0, 100, 10)

thermal_stress_index = st.slider("Thermal Stress Index", 0.0, 1.0, 0.3)
aging_indicator = st.slider("Aging Indicator", 0.0, 1.0, 0.4)

over_temp_flag = st.selectbox("Over Temperature Flag", [0, 1])
over_voltage_flag = st.selectbox("Over Voltage Flag", [0, 1])

# ------------------ PREDICTIONS ------------------

if st.button("🚀 Predict"):

    # ----- Linear Regression Prediction -----
    reg_input = np.array([[
        SOC, SOH, terminal_voltage, battery_current,
        battery_temp, ambient_temp, internal_resistance,
        charging_efficiency, cycle_degradation
    ]])

    reg_input_scaled = scaler.transform(reg_input)
    predicted_charging_time = lr_model.predict(reg_input_scaled)[0]

    # ----- Random Forest Classification -----
    clf_input = np.array([[
        SOC, SOH, battery_temp, ambient_temp,
        terminal_voltage, internal_resistance,
        thermal_stress_index, aging_indicator
    ]])

    trip_prediction = rf_model.predict(clf_input)[0]

    # ------------------ OUTPUT ------------------
    st.subheader("📊 Prediction Results")

    st.success(f"⏱ Estimated Charging Time: {predicted_charging_time:.2f} minutes")

    if trip_prediction == 1 and over_temp_flag == 0 and over_voltage_flag == 0:
        st.success("✅ Trip Feasible: Bus can safely complete the route")
    else:
        st.error("❌ Trip NOT Feasible: Risk of mid-route breakdown")

