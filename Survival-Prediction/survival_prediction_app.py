import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("Knn_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Survival Prediction App by Ashish Rajput")
st.markdown("Enter passenger details to predict survival:")

# Collect user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embark_town = st.selectbox("Embark Town", ["Southampton", "Cherbourg", "Queenstown"])
alone = st.selectbox("Alone", [True, False])

# Prediction button
if st.button("Predict"):

    # Raw input dictionary
    raw_input = {
        'pclass': pclass,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'alone': int(alone),

        # One-hot encoding
        'sex_' + sex: 1,
        'embark_town_' + embark_town: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Output result
    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")