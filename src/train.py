# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


# Load the California housing dataset and create a DataFrame
def load_data():
    housing = fetch_california_housing()
    # Create the matrix A
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    # Create the vector b
    df["Price"] = housing.target
    return df


# Train the model using linear regression and save it to a file using joblib
def train():
    df = load_data()
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score:.4f}")

    # Save the trained model to a file using joblib
    joblib.dump(model, "models/model.pkl")


if __name__ == "__main__":
    train()
