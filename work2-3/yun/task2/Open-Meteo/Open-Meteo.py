import requests
import json
import jsonpath
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 26.05942,
    "longitude": 119.198,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "weather_code",
        "cloudcover",
        "wind_speed_10m",
        "wind_direction_10m",
        "shortwave_radiation_instant",
        "is_day",
    ],
    "daily": [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "sunshine_duration",
    ],
    "timezone": "auto",
}
response = requests.get(url, params=params)
if response.status_code == 200:
    date=jsonpath.jsonpath(json.loads(response.text), "$.daily.time")
    temperature_2m_mean=jsonpath.jsonpath(json.loads(response.text), "$.daily.temperature_2m_mean")#平均气温
    temperature_2m_max=jsonpath.jsonpath(json.loads(response.text), "$.daily.temperature_2m_max")#最高气温
    temperature_2m_min=jsonpath.jsonpath(json.loads(response.text), "$.daily.temperature_2m_min")#最低气温
    precipitation_sum=jsonpath.jsonpath(json.loads(response.text), "$.daily.precipitation_sum")#降水量
    sunshine_duration=jsonpath.jsonpath(json.loads(response.text), "$.daily.sunshine_duration")#日照时数
    hour=jsonpath.jsonpath(json.loads(response.text), "$.hourly.time")[0]#小时
    temperature_2m=jsonpath.jsonpath(json.loads(response.text), "$.hourly.temperature_2m")[0]#小时温度
    relative_humidity_2m=jsonpath.jsonpath(json.loads(response.text), "$.hourly.relative_humidity_2m")[0]#小时相对湿度
    apparent_temperature=jsonpath.jsonpath(json.loads(response.text), "$.hourly.apparent_temperature")[0]#小时体感温度
    precipitation=jsonpath.jsonpath(json.loads(response.text), "$.hourly.precipitation")[0]#小时降水量
    weather_code=jsonpath.jsonpath(json.loads(response.text), "$.hourly.weather_code")[0]#小时天气代码
    cloudcover=jsonpath.jsonpath(json.loads(response.text), "$.hourly.cloudcover")[0]#小时云量
    wind_speed_10m=jsonpath.jsonpath(json.loads(response.text), "$.hourly.wind_speed_10m")[0]#小时10米风速
    wind_direction_10m=jsonpath.jsonpath(json.loads(response.text), "$.hourly.wind_direction_10m")[0]#小时10米风向
    shortwave_radiation_instant=jsonpath.jsonpath(json.loads(response.text), "$.hourly.shortwave_radiation_instant")[0]#小时短波辐射
    is_day=jsonpath.jsonpath(json.loads(response.text), "$.hourly.is_day")[0]#小时是否白天
    with open("weather_data.csv", "w", encoding="utf-8") as file:
        for i in range(len(date[0])):
            file.write(f"{date[0][i]},'平均温度：',{temperature_2m_mean[0][i]},'最高温度：',{temperature_2m_max[0][i]},'最低温度：',{temperature_2m_min[0][i]},'降水量：',{precipitation_sum[0][i]},'日照时数：',{sunshine_duration[0][i]}\n")
            for j in range(i*24,(i+1)*24):
                file.write(f"{hour[j]}\n")
                file.write(f"小时温度：{temperature_2m[j]}\n")
                file.write(f"小时相对湿度：{relative_humidity_2m[j]}\n")
                file.write(f"小时体感温度：{apparent_temperature[j]}\n")
                file.write(f"小时降水量：{precipitation[j]}\n")
                file.write(f"小时天气代码：{weather_code[j]}\n")
                file.write(f"小时云量：{cloudcover[j]}\n")
                file.write(f"小时10米风速：{wind_speed_10m[j]}\n")
                file.write(f"小时10米风向：{wind_direction_10m[j]}\n")
                file.write(f"小时短波辐射：{shortwave_radiation_instant[j]}\n")
                file.write(f"是否白天：{is_day[j]}\n")