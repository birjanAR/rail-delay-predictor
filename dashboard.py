import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("🚆 Rail Delay Predictor")
st.write("Predict whether a train is likely to be delayed")

# -----------------------
# USER INPUTS
# -----------------------

hour = st.slider("Hour of day", 0, 23, 12)
day_of_week = st.selectbox("Day of week (0=Mon, 6=Sun)", [0,1,2,3,4,5,6])
is_peak = st.selectbox("Is it peak time?", [0, 1])
weather_risk = st.selectbox("Weather delay risk", [0, 1, 2])

# -----------------------
# PREDICTION
# -----------------------

if st.button("Predict Delay"):
    features = np.array([[hour, day_of_week, is_peak, weather_risk]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Likely Delayed ({probability:.2%})")
    else:
        st.success(f"✅ Likely On Time ({probability:.2%})")