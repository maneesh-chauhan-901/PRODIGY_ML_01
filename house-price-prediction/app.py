import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import train_model, evaluate_model, get_model_coefficients

# Page configuration for a clean, modern layout
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏡",
    layout="wide",
)

st.title("House Price Prediction System")
st.write("Use the form below to predict house prices based on living area, bedrooms, and bathrooms.")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

# Load data and train model once
with st.spinner("Loading dataset and building the model..."):
    df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(df)
    y_pred = model.predict(X_test)
    mae, rmse, r2 = evaluate_model(y_test, y_pred)
    coef = get_model_coefficients(model)

# Columns layout for the main page
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("House Price Prediction")
    st.write("Enter the house details and click Predict Price to see the estimated sale price.")
    with st.form("price_form"):
        living_area = st.number_input(
            "House square footage (GrLivArea)",
            min_value=0,
            value=1500,
            step=50,
            help="Enter the total above-ground living area in square feet.",
        )
        bedrooms = st.number_input(
            "Number of bedrooms (BedroomAbvGr)",
            min_value=0,
            value=3,
            step=1,
            help="Enter the number of bedrooms above ground.",
        )
        bathrooms = st.number_input(
            "Number of bathrooms (FullBath)",
            min_value=0,
            value=2,
            step=1,
            help="Enter the number of full bathrooms.",
        )
        predict_button = st.form_submit_button("Predict Price")

    if predict_button:
        new_house = np.array([[living_area, bedrooms, bathrooms]])
        predicted_price = model.predict(new_house)[0]
        st.success(f"Estimated House Price: ${predicted_price:,.2f}")
        st.write("---")
        st.write("### Input details")
        st.write(f"- Living area: {living_area} sq ft")
        st.write(f"- Bedrooms: {bedrooms}")
        st.write(f"- Bathrooms: {bathrooms}")

with col2:
    st.subheader("Model Performance")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("MAE", f"${mae:,.2f}")
    st.metric("RMSE", f"${rmse:,.2f}")
    st.write("### Coefficients")
    st.write(coef)

st.write("---")

st.subheader("Data Visualization")
st.write("Visualizations help show how the model relates living area to price and how predictions compare to real values.")

fig1, ax1 = plt.subplots()
ax1.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)
ax1.set_xlabel("Living Area (sq ft)")
ax1.set_ylabel("Sale Price")
ax1.set_title("Living Area vs Sale Price")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual Price vs Predicted Price")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
errors = y_test - y_pred
ax3.hist(errors, bins=30, color="#2a7ae2", alpha=0.75)
ax3.set_xlabel("Prediction Error")
ax3.set_ylabel("Frequency")
ax3.set_title("Prediction Error Distribution")
st.pyplot(fig3)

st.write("---")

st.subheader("About the Model")
st.write(
    "This app uses a multiple linear regression model trained on the Ames housing dataset. "
    "The model uses living area, number of bedrooms, and number of bathrooms to estimate the sale price. "
    "The training process includes data cleaning, splitting into train/test sets, model training, and evaluation."
)

st.write("### Data preview")
st.dataframe(df[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]].head())
