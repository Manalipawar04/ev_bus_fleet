import streamlit as st
import numpy as np
import joblib

# Load trained models and scaler
trip_model = joblib.load("model_trip.pkl")
health_model = joblib.load("model_health.pkl")
charge_model = joblib.load("model_charge.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.set_page_config(page_title="EV Bus Fleet ML Platform", layout="centered")
st.title("🚍 EV Bus Fleet Analytics")

st.write(
    "This system predicts **Trip Feasibility**, **Battery Health (SOH)**, "
    "and **Charging Time** using historical EV battery telemetry."
)

# Sidebar inputs
st.sidebar.header("🔧 Input Battery Parameters")

SOC = st.sidebar.slider("State of Charge (SOC %)", 0, 100, 60)
terminal_voltage = st.sidebar.number_input("Terminal Voltage (V)", value=400.0)
battery_current = st.sidebar.number_input("Battery Current (A)", value=50.0)
battery_temp = st.sidebar.slider("Battery Temperature (°C)", 0, 80, 35)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 0, 60, 30)
internal_resistance = st.sidebar.number_input("Internal Resistance (Ohm)", value=0.02)
dT_dt = st.sidebar.number_input("dT/dt (Thermal Rate)", value=0.5)
dV_dt = st.sidebar.number_input("dV/dt (Voltage Rate)", value=0.3)
thermal_stress_index = st.sidebar.number_input("Thermal Stress Index", value=1.2)

# Prepare input array
input_data = np.array([[
    SOC,
    terminal_voltage,
    battery_current,
    battery_temp,
    ambient_temp,
    internal_resistance,
    dT_dt,
    dV_dt,
    thermal_stress_index
]])

input_scaled = scaler.transform(input_data)

# Prediction button
if st.button("🔍 Predict"):
    trip_pred = trip_model.predict(input_scaled)[0]
    health_pred = health_model.predict(input_scaled)[0]
    charge_pred = charge_model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Results")

    # Trip feasibility result
    if trip_pred == 1:
        st.success("✅ Trip Feasible – Bus can complete the route safely.")
    else:
        st.error("❌ Trip Not Feasible – Charging required before dispatch.")

    # Battery health
    st.info(f"🔋 Estimated Battery Health (SOH): **{health_pred:.2f}%**")

    # Charging time
    st.warning(f"⚡ Estimated Required Charging Time: **{charge_pred:.2f} minutes**")

# Footer
st.markdown("---")
st.caption("ML-based Decision Support System for EV Bus Fleet Operations")
