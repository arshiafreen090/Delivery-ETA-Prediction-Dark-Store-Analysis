import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Delivery ETA Prediction by Afreen",
    page_icon="🚴",
    layout="wide"
)


# CUSTOM THEME

st.markdown("""
<style>

    /* Primary button */
    .stButton > button {
        background-color: #34C363 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #2EAA56 !important;
        color: white !important;
        border: none !important;
    }

    /* Number input focus */
    .stNumberInput input:focus {
        border-color: #34C363 !important;
        box-shadow: 0 0 0 1px #34C363 !important;
    }

    /* Selectbox focus */
    [data-baseweb="select"] > div:focus-within {
        border-color: #34C363 !important;
        box-shadow: 0 0 0 1px #34C363 !important;
    }

    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #34C363 !important;
    }

</style>
""", unsafe_allow_html=True)

# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load("final_eta_model.pkl")


model = load_model()


# HEADER

st.title("Delivery ETA Prediction by Afreen")

st.markdown(
    "Enter the delivery details below to estimate the delivery time."
)

st.divider()


# DELIVERY PERSON

st.subheader("Delivery Person")

col1, col2, col3 = st.columns(3)

with col1:
    delivery_age = st.number_input(
        "Delivery Person Age",
        min_value=18,
        max_value=60,
        value=30
    )

with col2:
    delivery_rating = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

with col3:
    vehicle_condition = st.number_input(
        "Vehicle Condition",
        min_value=0,
        max_value=5,
        value=2
    )


# LOCATION

st.subheader("Location")

col1, col2 = st.columns(2)

with col1:
    restaurant_latitude = st.number_input(
        "Restaurant Latitude",
        value=12.9716,
        format="%.6f"
    )

    restaurant_longitude = st.number_input(
        "Restaurant Longitude",
        value=77.5946,
        format="%.6f"
    )

with col2:
    delivery_latitude = st.number_input(
        "Delivery Location Latitude",
        value=12.9352,
        format="%.6f"
    )

    delivery_longitude = st.number_input(
        "Delivery Location Longitude",
        value=77.6245,
        format="%.6f"
    )

distance_km = st.number_input(
    "Distance (km)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.1
)


# ORDER AND ROAD CONDITIONS

st.subheader("Order & Road Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    weather = st.selectbox(
        "Weather Conditions",
        [
            "Sunny",
            "Stormy",
            "Sandstorms",
            "Windy",
            "Cloudy",
            "Fog"
        ]
    )

with col2:
    traffic = st.selectbox(
        "Road Traffic Density",
        [
            "Low",
            "Medium",
            "High",
            "Jam"
        ]
    )

with col3:
    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )


# ORDER DETAILS

col1, col2, col3 = st.columns(3)

with col1:
    order_type = st.selectbox(
        "Type of Order",
        [
            "Snack",
            "Meal",
            "Drinks",
            "Buffet"
        ]
    )

with col2:
    vehicle_type = st.selectbox(
        "Type of Vehicle",
        [
            "motorcycle",
            "scooter",
            "electric_scooter",
            "bicycle"
        ]
    )

with col3:
    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [
            "0",
            "1",
            "2",
            "3"
        ]
    )


# TIME AND CITY

st.subheader("Time & City")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox(
        "City",
        [
            "Urban",
            "Metropolitian",
            "Semi-Urban"
        ]
    )

with col2:
    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

with col3:
    order_hour = st.slider(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=14
    )

picked_hour = st.slider(
    "Picked Hour",
    min_value=0,
    max_value=23,
    value=14
)


# PREDICTION

st.divider()

if st.button(
    "Predict Delivery Time",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame([{
        "Delivery_person_Age": delivery_age,
        "Delivery_person_Ratings": delivery_rating,
        "Restaurant_latitude": restaurant_latitude,
        "Restaurant_longitude": restaurant_longitude,
        "Delivery_location_latitude": delivery_latitude,
        "Delivery_location_longitude": delivery_longitude,
        "Weather_conditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival,
        "City": city,
        "Order_Hour": order_hour,
        "Picked_Hour": picked_hour,
        "Day_of_Week": day_of_week,
        "distance_km": distance_km
    }])

    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 1)

    st.success("Prediction completed!")

    st.metric(
        label="Estimated Delivery Time",
        value=f"{prediction} minutes"
    )