import requests
import json
import csv
import datetime

def change_time(time):
    dt = datetime.datetime.fromtimestamp(time + 28800)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

url='https://archive-api.open-meteo.com/v1/archive'
params = {
    "latitude": 26.05942,
    "longitude": 119.198,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,sunshine_duration,precipitation_sum",
    "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation_instant,is_day",
    "timezone": "auto",
    "format": "json",
    "timeformat": "unixtime"
}
headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
}
print("即将开始爬取......")
response=requests.get(url=url,params=params,headers=headers)
page_dic=response.json()

#处理小时数据
print("正在处理小时数据")
hour_dic=page_dic["hourly"]
length_hour=len(hour_dic["time"])
hour_time_str_list=[]
for i in range(length_hour):
    hour_time_str_list.append(change_time(hour_dic["time"][i]))
hour_list=[]
for i in range(length_hour):
    hour_list.append({"time_str":hour_time_str_list[i]})
    for key in hour_dic.keys():
        hour_list[i][key]=hour_dic[key][i]

print("正在保存小时数据")
with open("Hour_data.csv","w+",encoding="utf-8",newline='') as f:
    fieldnames=[
        "time_str","time","temperature_2m","relative_humidity_2m","apparent_temperature","precipitation","weather_code","cloud_cover","wind_speed_10m","wind_direction_10m","shortwave_radiation_instant","is_day"
    ]
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    writer.writerows(hour_list)

#处理日期数据
print("正在处理日期数据")
day_dic=page_dic["daily"]
length_day=len(day_dic["time"])
day_time_str_list=[]
for i in range(length_day):
    day_time_str_list.append(change_time(day_dic["time"][i]))
day_list=[]
for i in range(length_day):
    day_list.append({"time_str":day_time_str_list[i]})
    for key in day_dic.keys():
        day_list[i][key]=day_dic[key][i]

print("正在保存日期数据")
with open("Day_data.csv","w+",encoding="utf-8",newline='') as f:
    fieldnames=[
        "time_str","time","temperature_2m_mean","temperature_2m_max","temperature_2m_min","sunshine_duration","precipitation_sum"
    ]
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    writer.writerows(day_list)
print("\n爬取流程结束")