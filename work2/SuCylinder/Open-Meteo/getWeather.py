import requests
import pandas
from pathlib import Path
import json

api = "https://historical-forecast-api.open-meteo.com/v1/forecast"

params = {
	"latitude": 26.05942,
	"longitude": 119.198,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"daily": ["temperature_2m_max", "temperature_2m_mean", "temperature_2m_min", "precipitation_sum", "sunshine_duration"],
	"hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "cloud_cover", "weather_code", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation_instant", "is_day"],
}

response = requests.get(api,params=params)

data = response.json()

hourlyCsvPath = Path(__file__).parent / "hourly.csv"
hourlyLength = len(data["hourly"]["time"])

hourlyColums = [
    "date",
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",
    "is_day",
]
df_empty = pandas.DataFrame(columns=hourlyColums)
df_empty.to_csv(hourlyCsvPath, index=False, encoding="utf-8")

for i in range(0, hourlyLength):
    time = data["hourly"]["time"][i]
    temperature_2m = (
        str(data["hourly"]["temperature_2m"][i])
        + data["hourly_units"]["temperature_2m"]
    )
    relative_humidity_2m = (
        str(data["hourly"]["relative_humidity_2m"][i])
        + data["hourly_units"]["relative_humidity_2m"]
    )
    apparent_temperature = (
        str(data["hourly"]["apparent_temperature"][i])
        + data["hourly_units"]["apparent_temperature"]
    )
    precipitation = (
        str(data["hourly"]["precipitation"][i]) + data["hourly_units"]["precipitation"]
    )
    cloud_cover = (
        str(data["hourly"]["cloud_cover"][i]) + data["hourly_units"]["cloud_cover"]
    )
    weather_code = data["hourly"]["weather_code"][i]
    wind_speed_10m = (
        str(data["hourly"]["wind_speed_10m"][i])
        + data["hourly_units"]["wind_speed_10m"]
    )
    wind_direction_10m = (
        str(data["hourly"]["wind_direction_10m"][i])
        + data["hourly_units"]["wind_direction_10m"]
    )
    shortwave_radiation_instant = (
        str(data["hourly"]["shortwave_radiation_instant"][i])
        + data["hourly_units"]["shortwave_radiation_instant"]
    )
    is_day = data["hourly"]["is_day"][i]

    row = {
        "time": time,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "apparent_temperatur": apparent_temperature,
        "precipitation": precipitation,
        "cloud_cover": cloud_cover,
        "weather_code": weather_code,
        "wind_speed_10m": wind_speed_10m,
        "wind_direction_10m": wind_direction_10m,
        "shortwave_radiation_instant": shortwave_radiation_instant,
        "is_day": is_day,
    }
    df_row = pandas.DataFrame([row])
    df_row.to_csv(hourlyCsvPath, mode="a", header=False, index=False)


dailyCsvPath = Path(__file__).parent / "daily.csv"
dailyLength = len(data["daily"]["time"])

dailyColumn =[
    "time",
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
    "sunshine_duration"
]

df_empty = pandas.DataFrame(columns=dailyColumn)
df_empty.to_csv(dailyCsvPath, index=False, encoding="utf-8")

for i in range(0,dailyLength):
    time = str(data["daily"]["time"][i])
    temperature_2m_max = str(data["daily"]["temperature_2m_max"][i]) + data["daily_units"]["temperature_2m_max"]
    temperature_2m_mean = str(data["daily"]["temperature_2m_mean"][i]) + data["daily_units"]["temperature_2m_mean"]
    temperature_2m_min = str(data["daily"]["temperature_2m_min"][i]) + data["daily_units"]["temperature_2m_min"]
    precipitation_sum = str(data["daily"]["precipitation_sum"][i]) + data["daily_units"]["precipitation_sum"]
    sunshine_duration = str(data["daily"]["sunshine_duration"][i]) + data["daily_units"]["sunshine_duration"]

    row = {
        "time":time,
        "temperature_2m_max":temperature_2m_max,
        "temperature_2m_mean":temperature_2m_mean,
        "temperature_2m_min":temperature_2m_min,
        "precipitation_sum":precipitation_sum,
        "sunshine_duration":sunshine_duration
    }

    df_row = pandas.DataFrame([row])
    df_row.to_csv(dailyCsvPath,mode="a",header=False,index=False)