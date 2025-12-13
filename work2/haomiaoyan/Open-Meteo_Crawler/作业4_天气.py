# -*- coding: utf-8 -*-
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import requests

# 禁用代理
# 创建Session并禁用系统/环境变量代理
session = requests.Session()
session.trust_env = False  # 强制不走任何代理，直接访问服务器

# 优化缓存和重试配置（移除重复的status_forcelist参数）
# 缓存：1小时过期，避免永久缓存错误请求
cache_session = requests_cache.CachedSession(
    '.cache',
    expire_after=3600,  # 缓存有效期1小时
    session=session     # 复用禁用代理的Session
)

# 重试：简化参数，避免与底层重复（核心修复点）
retry_session = retry(
    cache_session,
    retries=3,          # 重试次数
    backoff_factor=0.5  # 重试间隔系数（0.5秒→1秒→2秒）
    # 移除status_forcelist参数，避免重复赋值
)

# 初始化Open-Meteo客户端
openmeteo = openmeteo_requests.Client(session=retry_session)

# 天气请求参数
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 26.05942,
    "longitude": 119.198,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "daily": [
        "weather_code", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "sunshine_duration"
    ],
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature",
        "dew_point_2m", "precipitation", "weather_code", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "is_day", "shortwave_radiation"
    ],
    "timezone": "Asia/Shanghai"  # 指定时区，方便本地时间查看
}

# 发送请求（添加异常处理）
try:
    responses = openmeteo.weather_api(url, params=params)
except Exception as e:
    print(f"请求失败：{e}")
    exit()

# 处理响应
# 处理第一个位置的天气数据
response = responses[0]
print(f"坐标：{response.Latitude()}°N {response.Longitude()}°E")
print(f"海拔：{response.Elevation()} m")
print(f"时区偏移：{response.UtcOffsetSeconds() / 3600} 小时")

# 处理小时数据
hourly = response.Hourly()
hourly_vars = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "dew_point_2m", "precipitation", "weather_code", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "is_day", "shortwave_radiation"
]
hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}
# 批量赋值小时数据
for i, var in enumerate(hourly_vars):
    hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

hourly_dataframe = pd.DataFrame(hourly_data)
# 转换时区为本地（Asia/Shanghai）
hourly_dataframe["date"] = hourly_dataframe["date"].dt.tz_convert("Asia/Shanghai")
print("\n小时数据预览：")
print(hourly_dataframe.head())

# 处理日数据
daily = response.Daily()
daily_vars = [
    "weather_code", "temperature_2m_mean", "temperature_2m_max",
    "temperature_2m_min", "precipitation_sum", "sunshine_duration"
]
daily_data = {"date": pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
)}
# 批量赋值日数据
for i, var in enumerate(daily_vars):
    daily_data[var] = daily.Variables(i).ValuesAsNumpy()

daily_dataframe = pd.DataFrame(daily_data)
# 转换时区为本地
daily_dataframe["date"] = daily_dataframe["date"].dt.tz_convert("Asia/Shanghai")
print("\n日数据预览：")
print(daily_dataframe.head())

# 保存数据
hourly_dataframe.to_csv("hourly_weather.csv", index=False, encoding="utf-8-sig")
daily_dataframe.to_csv("daily_weather.csv", index=False, encoding="utf-8-sig")
print("\n数据已保存到 hourly_weather.csv 和 daily_weather.csv")
