import requests
import pandas as pd
api=r"https://archive-api.open-meteo.com/v1/archive?latitude=26.05942&longitude=119.198&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_mean,temperature_2m_min,precipitation_sum,sunshine_duration&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation_instant,is_day"
r=requests.get(api)
results=r.json()
df=pd.DataFrame(results)
df.to_csv("data.csv")