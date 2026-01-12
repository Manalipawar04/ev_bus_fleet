import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SET UP PAGE ---
st.set_page_config(page_title="EV Bus Battery Intelligence", layout="wide")
st.title("⚡ EV Bus Battery Intelligence System")
st.markdown("""
Predict **State of Charge (SoC)**, **State of Health (SoH)**, and **Operational Risk** using real-time telemetry data.
""")

# --- 2. LOAD TRAINED MODELS ---
# Replace 'model.pkl' with the actual filename from your workflow
@st.cache_resource
def load_models():
    try:
        # In a real scenario, you'd load the .pkl files saved during training
        # model = joblib.load('ev_battery_model.pkl')
        # For this example, we assume the model logic is ready to use
        return None 
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_models()

# --- 3. SIDEBAR INPUTS (Telemetery Features) ---
st.sidebar.header("Real-time Telemetry Inputs")

def user_input_features():
    voltage = st.sidebar.slider("Voltage (V)", 300.0, 800.0, 600.0)
    current = st.sidebar.slider("Current (A)", -200.0, 200.0, 50.0)
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 60.0, 25.0)
    cycle_count = st.sidebar.number_input("Cycle Count", min_value=0, value=500)
    avg_speed = st.sidebar.slider("Average Trip Speed (km/h)", 0, 80, 35)
    
    data = {
        'Voltage': voltage,
        'Current': current,
        'Temperature': temp,
        'Cycle_Count': cycle_count,
        'Avg_Speed': avg_speed
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- 4. DISPLAY INPUT DATA ---
st.subheader("Current Telemetry Snapshot")
st.write(input_df)

# --- 5. PREDICTIONS & LOGIC ---
st.subheader("System Predictions")

# Dummy logic for demonstration based on the workflow provided
# In production, replace these with model.predict(input_df)
soc_pred = 85.5 - (input_df['Current'][0] * 0.05) # Simulated SoC
soh_pred = 98.0 - (input_df['Cycle_Count'][0] * 0.005) # Simulated SoH

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("State of Charge (SoC)", f"{soc_pred:.1f}%")
    st.progress(soc_pred / 100)

with col2:
    st.metric("State of Health (SoH)", f"{soh_pred:.1f}%")
    st.info("Aging status: Normal")

with col3:
    risk = "High" if temp > 50 or soc_pred < 15 else "Low"
    st.metric("Operational Risk", risk)
    if risk == "High":
        st.warning("⚠️ Warning: Risk of breakdown detected!")
    else:
        st.success("✅ System Reliable")

# --- 6. TRIP FEASIBILITY ---
st.subheader("Route Analysis")
required_soc = 20.0 # Example requirement for the next route
if soc_pred > required_soc:
    st.write(f"The bus has enough charge for the next fixed route (Requires {required_soc}%).")
else:
    st.error("Insufficient charge for next trip. Schedule immediate charging.")
