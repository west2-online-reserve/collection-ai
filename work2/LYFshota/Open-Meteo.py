import requests
import pathlib
import json
import csv
import re
import urllib.parse
import pandas as pd
from bs4 import BeautifulSoup

if __name__ == "__main__":
    api='https://archive-api.open-meteo.com/v1/archive?latitude=26.05942&longitude=119.198&start_date=2024-01-01&end_date=2024-12-31&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,daylight_duration,precipitation_sum&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_direction_10m,is_day,shortwave_radiation_instant&timezone=auto'
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    out_path=pathlib.Path(__file__).parent /"福州大学天气数据"
    out_path.mkdir(exist_ok=True)
    hourly_csv_path=out_path /"福州大学2024年1月-2024年12月每小时天气数据.csv"
    daily_csv_path=out_path /"福州大学2024年1月-2024年12月每日天气数据.csv"
    hourly_weather_dataframe=pd.DataFrame(columns=['温度(2m)(°C)','相对湿度(2m)(%)','风速(10m)(m/s)','体感温度(°C)','降水量(mm)','天气代码','云量(%)','风速(10m)(m/s)','风向(10m)(°)','短波辐射(W/m²)','白天/夜晚'])
    daily_weather_dataframe=pd.DataFrame(columns=['平均温度(2m)(°C)','最高温度(2m)(°C)','最低温度(2m)(°C)','降水量(mm)','日照时长(分钟)'])

    response=requests.get(url=api,headers=headers)
    response.encoding ='utf-8'
    weather_data=response.json()
    #处理每小时天气数据
    hourly_data=weather_data.get('hourly',{})
    hourly_time=hourly_data.get('time',[])
    hourly_temperature_2m=hourly_data.get('temperature_2m',[])
    hourly_relative_humidity_2m=hourly_data.get('relative_humidity_2m',[])
    hourly_wind_speed_10m=hourly_data.get('wind_speed_10m',[])
    hourly_apparent_temperature=hourly_data.get('apparent_temperature',[])
    hourly_precipitation=hourly_data.get('precipitation',[])
    hourly_weather_code=hourly_data.get('weather_code',[])
    hourly_cloud_cover=hourly_data.get('cloud_cover',[])
    hourly_wind_direction_10m=hourly_data.get('wind_direction_10m',[])
    hourly_is_day=hourly_data.get('is_day',[])
    hourly_shortwave_radiation_instant=hourly_data.get('shortwave_radiation_instant',[])
    for i in range(len(hourly_time)):
        hourly_weather_dataframe.loc[hourly_time[i]]=[
            hourly_temperature_2m[i],
            hourly_relative_humidity_2m[i],
            hourly_wind_speed_10m[i],
            hourly_apparent_temperature[i],
            hourly_precipitation[i],
            hourly_weather_code[i],
            hourly_cloud_cover[i],
            hourly_wind_speed_10m[i],
            hourly_wind_direction_10m[i],
            hourly_shortwave_radiation_instant[i],
            '白天' if hourly_is_day[i]==1 else '夜晚'
        ]
    hourly_weather_dataframe.index.name='时间'
    hourly_weather_dataframe.to_csv(hourly_csv_path,encoding='utf-8-sig')
    print(f"已保存每小时天气数据到 {hourly_csv_path}")
    #处理每日天气数据
    daily_data=weather_data.get('daily',{})
    daily_time=daily_data.get('time',[])
    daily_temperature_2m_mean=daily_data.get('temperature_2m_mean',[])
    daily_temperature_2m_max=daily_data.get('temperature_2m_max',[])
    daily_temperature_2m_min=daily_data.get('temperature_2m_min',[])
    daily_precipitation_sum=daily_data.get('precipitation_sum',[])
    daily_daylight_duration=daily_data.get('daylight_duration',[])
    for i in range(len(daily_time)):
        daily_weather_dataframe.loc[daily_time[i]]=[
            daily_temperature_2m_mean[i],
            daily_temperature_2m_max[i],
            daily_temperature_2m_min[i],
            daily_precipitation_sum[i],
            daily_daylight_duration[i]/60  #转换为小时
        ]


    daily_weather_dataframe.index.name='时间'
    daily_weather_dataframe.to_csv(daily_csv_path,encoding='utf-8-sig')
    print(f"已保存每日天气数据到 {daily_csv_path}")
