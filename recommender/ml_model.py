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
    Get monthly average temperature and humidity,
    plus estimated rainfall over a three-month period.

    Rainfall period: selected month and following two months.
    This is an approximation, not a crop-specific growing season.
    """
    import requests
    from datetime import datetime
    from statistics import mean

    month_numbers = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    month = month.upper()

    if month not in month_numbers:
        raise ValueError("Invalid month selected.")

    target_month = month_numbers[month]

    # Get average temperature and humidity for the selected month.
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

    # Retrieve daily rainfall for historical three-month periods.
    archive_url = "https://archive-api.open-meteo.com/v1/archive"

    archive_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2016-01-01",
        "end_date": "2026-02-28",
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
        if rainfall is None:
            continue

        date = datetime.fromisoformat(date_text)
        key = (date.year, date.month)
        monthly_totals[key] = (
            monthly_totals.get(key, 0.0) + rainfall
        )

    # Calculate three-month totals for seasons starting in 2016–2025.
    seasonal_totals = []

    for start_year in range(2016, 2026):
        total = 0.0
        complete = True

        for offset in range(3):
            absolute_month = target_month + offset
            year = start_year + (absolute_month - 1) // 12
            calendar_month = (absolute_month - 1) % 12 + 1
            key = (year, calendar_month)

            if key not in monthly_totals:
                complete = False
                break

            total += monthly_totals[key]

        if complete:
            seasonal_totals.append(total)

    if len(seasonal_totals) < 8:
        raise ValueError(
            "Not enough historical rainfall data for this location."
        )

    seasonal_rainfall = round(mean(seasonal_totals), 2)

    return temperature, humidity, seasonal_rainfall

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