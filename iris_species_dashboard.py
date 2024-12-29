import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Model ko load karna
model = joblib.load("decision_tree_model.pkl")

# Feature normalization ke liye scaler load karna
scaler = MinMaxScaler()

# Streamlit app banani hai
def interactive_dashboard():
    st.title("Iris Species Prediction")

    # User se input lena
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1)
    petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "SepalLengthCm": [sepal_length],
            "SepalWidthCm": [sepal_width],
            "PetalLengthCm": [petal_length],
            "PetalWidthCm": [petal_width]
        })

        # Features ko normalize karna
        input_data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] = scaler.fit_transform(input_data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])

        # Prediction karna
        prediction = model.predict(input_data)
        st.write(f"Predicted Species: {prediction[0]}")

interactive_dashboard()
