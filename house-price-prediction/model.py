import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path: str = "train.csv") -> pd.DataFrame:
    """Load the house dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """Select required columns and drop missing values."""
    features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
    target = "SalePrice"
    data = df[features + [target]].dropna()
    return data


def train_model(df: pd.DataFrame):
    """Split the dataset and train a linear regression model."""
    data = preprocess_data(df)
    X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics for regression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def get_model_coefficients(model: LinearRegression):
    """Return model coefficients in a readable format."""
    coefficients = {
        "GrLivArea": model.coef_[0],
        "BedroomAbvGr": model.coef_[1],
        "FullBath": model.coef_[2],
    }
    return coefficients
