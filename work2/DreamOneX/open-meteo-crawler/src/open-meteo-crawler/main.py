import requests
import pandas as pd
from typing import Any

params = {
    "latitude": 26.05942,
    "longitude": 119.198,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "daily": "precipitation_sum,sunshine_duration,temperature_2m_min,temperature_2m_max,temperature_2m_mean",
    "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,is_day,shortwave_radiation"
}

response: requests.Response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive", params=params)

if response.status_code == 200:
    data: dict[str, Any] = response.json()
    
    # Process Daily Data
    if "daily" in data:
        daily_df = pd.DataFrame(data["daily"])
        daily_df.to_csv("daily.csv", index=False)
        print("Saved daily.csv")
        
    # Process Hourly Data
    if "hourly" in data:
        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df.to_csv("hourly.csv", index=False)
        print("Saved hourly.csv")
else:
    print(f"Error: {response.status_code}")
    print(response.text)