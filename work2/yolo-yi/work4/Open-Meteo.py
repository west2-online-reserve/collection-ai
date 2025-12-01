import requests

base_url = "https://archive-api.open-meteo.com/v1/archive"

hourly_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation_instant",
    "is_day"
]

daily_vars = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "sunshine_duration"
]

query_string = (
    f"?latitude=26.05942"
    f"&longitude=119.198"
    f"&start_date=2024-01-01"
    f"&end_date=2024-12-31"
    f"&timezone=Asia/Shanghai"
    f"&format=csv"  
    f"&hourly={','.join(hourly_vars)}"  
    f"&daily={','.join(daily_vars)}"
)

full_url = base_url + query_string

# 发起请求
response = requests.get(full_url)

# 检查并保存
if response.status_code == 200:
    filename = "fzu_weather_2024.csv"
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"成功！数据已保存为: {filename}")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)
