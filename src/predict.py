import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
custom_house = pd.DataFrame(
    {
        "MedInc": [3.5],
        "HouseAge": [20],
        "AveRooms": [5],
        "AveBedrms": [1],
        "Population": [1000],
        "AveOccup": [3],
        "Latitude": [34.05],
        "Longitude": [-118.25],
    }
)
prediction = model.predict(custom_house)
print(f"Prediction: {prediction}")
