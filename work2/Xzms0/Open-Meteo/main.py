from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


ROOT_DIR = Path(__file__).absolute().parent
URL = "https://archive-api.open-meteo.com/v1/archive"

# Make sure all required weather variables are listed here
PARAMS = {
        "latitude": 26.05942,
        "longitude": 119.198,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": ["temperature_2m_max", "temperature_2m_min", 
                "precipitation_sum", "sunshine_duration", 
                "temperature_2m_mean"],
        "hourly": ["temperature_2m", "relative_humidity_2m", 
                "precipitation", "apparent_temperature", 
                "cloud_cover", "wind_speed_10m", 
                "wind_direction_10m", "shortwave_radiation", 
                "is_day"]}


def hourly(response):
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()
    hourly_shortwave_radiation = hourly.Variables(7).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(8).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
    hourly_data["is_day"] = hourly_is_day

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe


def daily(response):
    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(4).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean

    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe


def main():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # The order of variables in hourly or daily is important to assign them correctly below
    responses = openmeteo.weather_api(URL, params=PARAMS)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    hourly_dataframe = hourly(response)
    hourly_dataframe.to_csv(ROOT_DIR/"hourly_data.csv",encoding='utf-8',index=False)
    daily_dataframe = daily(response)
    daily_dataframe.to_csv(ROOT_DIR/"daily_data.csv",encoding='utf-8',index=False)


if __name__ == "__main__":
    main()
