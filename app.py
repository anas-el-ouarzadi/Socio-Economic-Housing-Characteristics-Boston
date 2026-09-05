# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the saved Ridge pipeline ---
model = joblib.load("ridge_model.pkl")

# --- Preprocessing function ---
def preprocess_single_input(crim, zn, lstat, rm, ptratio):
    crim_log = np.log1p(crim)
    zn_log = np.log1p(zn)
    lstat_log = np.log1p(lstat)
    rm_winsor = rm if rm <= 7.5 else 7.5
    return np.array([[lstat_log, rm_winsor, zn_log, ptratio, crim_log]])

# --- Dashboard Title ---
st.title("🏠 Boston Housing Value Prediction (Dynamic Interaction)")

st.write("Adjust the sliders below — prediction and plots update instantly.")

# --- User Inputs (dynamic widgets) ---
st.header("Socio‑Economic Factors")
crim = st.slider("Crime rate (CRIM)", 0.0, 1.0, 0.03, 0.01)
zn = st.slider("Residential land proportion (ZN)", 0.0, 30.0, 18.0, 1.0)
lstat = st.slider("Lower status population % (LSTAT)", 0.0, 40.0, 12.0, 0.5)
ptratio = st.slider("Pupil-teacher ratio (PTRATIO)", 10.0, 25.0, 15.0, 0.1)

st.header("Housing Characteristics")
rm = st.slider("Average number of rooms (RM)", 3.0, 9.0, 6.0, 0.1)

# --- Prediction (auto updates) ---
X_input = preprocess_single_input(crim, zn, lstat, rm, ptratio)
raw_prediction = model.predict(X_input)[0]

if raw_prediction <= 0:
    st.warning("⚠️ Model predicted zero or negative value — inputs may be outside training range.")
    prediction = 0.0
else:
    prediction = raw_prediction

st.success(f"🏡 Predicted Median Value of Home: {prediction:.2f} (in $1000s)")

# --- Visualization by categories ---
st.write("### 📊 Socio‑Economic Inputs")
fig1, ax1 = plt.subplots(figsize=(8,5))
features_socio = ["CRIM", "ZN", "LSTAT", "PTRATIO"]
values_socio = [crim, zn, lstat, ptratio]
sns.barplot(x=features_socio, y=values_socio, palette="Blues", ax=ax1)
ax1.set_ylabel("Value")
ax1.set_title("Socio‑Economic Factors")
st.pyplot(fig1)

st.write("### 🏠 Housing Characteristics")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x=["RM"], y=[rm], palette="Greens", ax=ax2)
ax2.set_ylabel("Value")
ax2.set_title("Housing Characteristic")
st.pyplot(fig2)

# --- Sensitivity Analysis ---
st.write("### 🔎 Sensitivity Analysis: Effect of Rooms (RM)")
rm_range = np.linspace(4, 9, 50)
preds = [max(0, model.predict(preprocess_single_input(crim, zn, lstat, rm_val, ptratio))[0]) for rm_val in rm_range]
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(rm_range, preds, color="darkorange", linewidth=2)
ax3.axvline(rm, color="red", linestyle="--", label="Your Input RM")
ax3.set_xlabel("Average Rooms (RM)")
ax3.set_ylabel("Predicted MEDV ($1000s)")
ax3.set_title("Impact of Rooms on Predicted Value")
ax3.legend()
st.pyplot(fig3)
