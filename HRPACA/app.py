import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load trained model
model = pickle.load(open("model/readmission_model.pkl", "rb"))

st.title("Hospital Readmission Prediction Dashboard")

# Load cleaned data
df = pd.read_csv("data/cleaned_hospital_data.csv")

# Print column names to verify
print(df.columns)

# Rename columns only if they exist
if "Number of Discharges" in df.columns:
    df.rename(columns={"Number of Discharges": "num_discharges"}, inplace=True)

if "Expected Readmission Rate" not in df.columns and "Predicted Readmission Rate" in df.columns:
    df.rename(columns={"Predicted Readmission Rate": "Expected Readmission Rate"}, inplace=True)

# Show dataset
if st.checkbox("Show Sample Data"):
    st.write(df.head())

# User Input
num_discharges = st.slider("Number of Discharges", 10, 5000, 500)
expected_readmission_rate = st.slider("Expected Readmission Rate", 5.0, 25.0, 10.0)

# Predict Readmission Risk
if st.button("Predict"):
    input_features = pd.DataFrame([[num_discharges, expected_readmission_rate]], columns=['num_discharges', 'Expected Readmission Rate'])
    input_features = input_features.reindex(columns=model.feature_names_in_, fill_value=0)
    predicted_ratio = model.predict(input_features)[0]
    st.write(f"Predicted Readmission Ratio: {predicted_ratio:.2f}")

# Plot Readmission Rate Distribution
st.subheader("Readmission Rate Distribution")
fig, ax = plt.subplots()
df["Expected Readmission Rate"].hist(bins=30, ax=ax)
ax.set_xlabel("Readmission Rate")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Live Graph for Prediction
st.subheader("Live Prediction Graph")

fig, ax = plt.subplots()

if "num_discharges" in df.columns and "Expected Readmission Rate" in df.columns:
    scatter = ax.scatter(df["num_discharges"], df["Expected Readmission Rate"], alpha=0.5)

    def update_plot(num_discharges, expected_readmission_rate):
        input_features = pd.DataFrame([[num_discharges, expected_readmission_rate]], columns=["num_discharges", "Expected Readmission Rate"])
        input_features = input_features.reindex(columns=model.feature_names_in_, fill_value=0)
        predicted_ratio = model.predict(input_features)[0]
        scatter.set_offsets([[num_discharges, predicted_ratio]])
        ax.set_xlim(0, df["num_discharges"].max())
        ax.set_ylim(0, df["Expected Readmission Rate"].max())
        st.pyplot(fig)

    st.slider("Adjust Number of Discharges", 10, 5000, 500, key="num_discharges_slider", on_change=lambda: update_plot(st.session_state.num_discharges_slider, expected_readmission_rate))
    st.slider("Adjust Expected Readmission Rate", 5.0, 25.0, 10.0, key="expected_readmission_rate_slider", on_change=lambda: update_plot(num_discharges, st.session_state.expected_readmission_rate_slider))
else:
    st.error("Missing required columns for visualization.")

