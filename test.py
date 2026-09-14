import joblib
import pandas as pd

model = joblib.load("final_eta_model.pkl")

print("Model loaded:", type(model))

sample = pd.DataFrame([{
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.8,
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Low",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": "1",
    "Festival": "No",
    "City": "Metropolitian",
    "Order_Hour": 14,
    "Picked_Hour": 14,
    "Day_of_Week": "Wednesday",
    "distance_km": 5.0
}])

prediction = model.predict(sample)

print("Prediction:", prediction)