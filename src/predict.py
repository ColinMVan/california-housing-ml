import joblib

model = joblib.load("models/model.pkl")
example = [[8.3252, 41.0, 6.984127, 1.0238095, 322.0, 2.555556, 37.88, -122.23]]
prediction = model.predict(example)
print(f"Prediction: {prediction}")
