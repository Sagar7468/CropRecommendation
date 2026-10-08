import os
import joblib
import pandas as pd
import requests


# Load trained ML model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ml_models",
    "crop_recommendation_model.pkl"
)

model = joblib.load(MODEL_PATH)


def get_climate_data(latitude, longitude, month):
    """
    Get climate data from NASA POWER API.
    """

    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    temperature = data["properties"]["parameter"]["T2M"][month]
    humidity = data["properties"]["parameter"]["RH2M"][month]
    precipitation = data["properties"]["parameter"]["PRECTOTCORR"][month]

    monthly_rainfall = precipitation * 30

    return temperature, humidity, monthly_rainfall


def recommend_crops(
    latitude,
    longitude,
    N,
    P,
    K,
    ph,
    month="JUN",
    temperature_change=2.0,
    rainfall_change_percent=-10.0
):
    """
    Generate crop recommendations using
    NASA climate data and the trained ML model.
    """

    # Get current climate data
    temperature, humidity, rainfall = get_climate_data(
        latitude,
        longitude,
        month
    )

    # Apply climate-change scenario
    future_temperature = temperature + temperature_change

    future_rainfall = rainfall * (
        1 + rainfall_change_percent / 100
    )

    # Prepare input for ML model
    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": future_temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": future_rainfall
    }])

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    crop_names = model.classes_

    crop_probabilities = sorted(
        zip(crop_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 3 crops
    top_3 = crop_probabilities[:3]

    return {
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "rainfall": round(rainfall, 2),
        "future_temperature": round(future_temperature, 2),
        "future_rainfall": round(future_rainfall, 2),
        "top_crops": [
            {
                "crop": crop,
                "probability": round(probability * 100, 2)
            }
            for crop, probability in top_3
        ]
    }