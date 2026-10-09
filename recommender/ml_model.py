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
    """Get climate averages and historical monthly rainfall."""

    import requests
    from datetime import datetime
    from statistics import mean

    # Get temperature and humidity from NASA POWER climatology.
    nasa_url = (
        "https://power.larc.nasa.gov/api/temporal/"
        "climatology/point"
    )

    nasa_params = {
        "parameters": "T2M,RH2M",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
    }

    response = requests.get(
        nasa_url, params=nasa_params, timeout=30
    )
    response.raise_for_status()
    nasa_data = response.json()["properties"]["parameter"]

    temperature = nasa_data["T2M"][month]
    humidity = nasa_data["RH2M"][month]

    # Average the total rainfall for this month across 2016–2025.
    month_numbers = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    if month not in month_numbers:
        raise ValueError("Invalid month selected.")

    target_month = month_numbers[month]

    archive_url = "https://archive-api.open-meteo.com/v1/archive"

    archive_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2016-01-01",
        "end_date": "2025-12-31",
        "daily": "precipitation_sum",
        "timezone": "auto",
    }

    response = requests.get(
        archive_url, params=archive_params, timeout=60
    )
    response.raise_for_status()

    daily_data = response.json()["daily"]
    monthly_totals = {}

    for date_text, rainfall in zip(
        daily_data["time"],
        daily_data["precipitation_sum"],
    ):
        date = datetime.fromisoformat(date_text)

        if date.month == target_month and rainfall is not None:
            monthly_totals.setdefault(date.year, []).append(rainfall)

    # Sum daily rainfall for each selected month, year by year.
    yearly_rainfall = [
        sum(values)
        for values in monthly_totals.values()
        if values
    ]

    if len(yearly_rainfall) < 8:
        raise ValueError(
            "Not enough historical rainfall data for this location."
        )

    monthly_rainfall = round(mean(yearly_rainfall), 2)

    return temperature, humidity, monthly_rainfall

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    temperature = data["properties"]["parameter"]["T2M"][month]
    humidity = data["properties"]["parameter"]["RH2M"][month]
    precipitation = data["properties"]["parameter"]["PRECTOTCORR"][month]

    monthly_rainfall = precipitation * 30

    return temperature, humidity, monthly_rainfall


def get_weather_forecast(latitude, longitude):
    """Fetch current weather and a 7-day forecast from Open-Meteo."""
    import requests

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max"
        ),
        "forecast_days": 7,
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        current = data["current"]
        daily = data["daily"]

        forecast = []

        for i in range(len(daily["time"])):
            forecast.append({
                "date": daily["time"][i],
                "max_temp": daily["temperature_2m_max"][i],
                "min_temp": daily["temperature_2m_min"][i],
                "rainfall": daily["precipitation_sum"][i],
                "rain_probability":
                    daily["precipitation_probability_max"][i]
            })

        return {
            "available": True,
            "current_temperature": current["temperature_2m"],
            "current_humidity": current["relative_humidity_2m"],
            "current_rainfall": current["precipitation"],
            "wind_speed": current["wind_speed_10m"],
            "forecast": forecast
        }

    except (requests.RequestException, KeyError, TypeError, ValueError):
        return {
            "available": False,
            "forecast": []
        }

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