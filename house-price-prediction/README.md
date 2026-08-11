# House Price Prediction System

This project demonstrates a simple house price prediction app built with Python, Scikit-learn, and Streamlit.

## Project Structure

```
house-price-prediction/
├── app.py
├── model.py
├── train.csv
├── requirements.txt
└── README.md
```

## Features

- Loads the house price dataset from `train.csv`
- Uses `GrLivArea`, `BedroomAbvGr`, and `FullBath` to predict `SalePrice`
- Performs data preprocessing, model training, and evaluation
- Displays MAE, RMSE, and R² score
- Shows visualizations for feature relationships and prediction performance
- Provides a clean Streamlit UI for entering house details and predicting price

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The model training logic is separated into `model.py`
- The Streamlit frontend is in `app.py`
- The dataset should remain in the same folder as `app.py`
