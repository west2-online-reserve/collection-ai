import time
import datetime
import requests
import json
import csv
from pprint import pprint
from collections import OrderedDict  #有序字典

headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
}

params = {
    'latitude':26.05942,
    'longitude': 119.198,
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'daily': ['temperature_2m_max','sunshine_duration','temperature_2m_mean','temperature_2m_min','precipitation_sum'],
    'hourly':['temperature_2m','relative_humidity_2m','precipitation','apparent_temperature','weather_code','wind_speed_10m','cloud_cover','wind_direction_10m','shortwave_radiation_instant','is_day'],
    'timeformat':'unixtime'
} 

def time_turned(timestamp):   #将时间戳转化为本地时间字符串
    #将Unix数组中的时间戳转化为UTC时间，用datetime模块的fromtimestamp功能
    utc_time = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    #用astimezone功能转换到本地时区即北京时间，UTC+8，即比UTC时间快8个小时，用timezone功能获得时区，用timedelta功能获得时间差
    local_time = utc_time.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    #转化为字符串（如：2024-01-01 08:00:00）
    time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return time_str
def tuple_remove(data): #去除因为有序字典产生的额外括号
    if isinstance(data,tuple):
        return data[0]
    else:
        return data

weather_url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'

response = requests.get(weather_url,params = params,headers = headers)
response_json = json.loads(response.text)

daily_timestamps = response_json['daily']['time']
daily_time = [time_turned(daily_timestamp) for daily_timestamp in daily_timestamps]

hourly_timestamps = response_json['hourly']['time']
hourly_time = [time_turned(hourly_timestamp) for hourly_timestamp in hourly_timestamps] 

temperature_2m_max = response_json['daily']['temperature_2m_max']
temperature_2m_mean = response_json['daily']['temperature_2m_mean']
temperature_2m_min = response_json['daily']['temperature_2m_min']
precipitation_sum = response_json['daily']['precipitation_sum']
sunshine_duration = response_json['daily']['sunshine_duration']

temperature_2m = response_json['hourly']['temperature_2m']
relative_humidity_2m = response_json['hourly']['relative_humidity_2m']
precipitation = response_json['hourly']['precipitation']
apparent_temperature = response_json['hourly']['apparent_temperature']
weather_code = response_json['hourly']['weather_code']
wind_speed_10m = response_json['hourly']['wind_speed_10m']
cloud_cover = response_json['hourly']['cloud_cover']
wind_direction_10m = response_json['hourly']['wind_direction_10m']
shortwave_radiation_instant = response_json['hourly']['shortwave_radiation_instant']
is_day = response_json['hourly']['is_day']

daily_datas = []
for i in range(0,len(daily_time)):
    daily_data = OrderedDict()  #创建有序字典对象
    daily_data['日期'] = tuple_remove(daily_time[i])
    daily_data['每日最高温度-2m'] = tuple_remove(temperature_2m_max[i])
    daily_data['每日平均温度-2m'] = tuple_remove(temperature_2m_mean[i])
    daily_data['每日最低温度-2m'] = tuple_remove(temperature_2m_min[i])
    daily_data['每日总降水量'] = tuple_remove(precipitation_sum[i])
    daily_data['每日光照时长'] = tuple_remove(sunshine_duration[i])
    daily_datas.append(daily_data)

with open(r'C:\Users\Lst12\Desktop\每日天气数据.csv', 'w', encoding='utf-8-sig', newline='') as f:
            headers = ['日期', '每日最高温度-2m', '每日平均温度-2m','每日最低温度-2m','每日总降水量','每日光照时长']
            writer = csv.DictWriter(f,headers)
            writer.writeheader()
            writer.writerows(daily_datas)


hourly_datas = []
for j in range(0,len(hourly_time)):
    hourly_data = OrderedDict()
    hourly_data['时刻'] = tuple_remove(hourly_time[j])
    hourly_data['当前温度-2m'] = tuple_remove(temperature_2m[j])
    hourly_data['当前相对湿度-2m'] = tuple_remove(relative_humidity_2m[j])
    hourly_data['当前降水量'] = tuple_remove(precipitation[j])
    hourly_data['当前体感温度'] = tuple_remove(apparent_temperature[j])
    hourly_data['天气编码'] = tuple_remove(weather_code[j])
    hourly_data['当前风速-10m'] = tuple_remove(wind_speed_10m[j])
    hourly_data['当前总云量'] = tuple_remove(cloud_cover[j])
    hourly_data['当前风向'] = tuple_remove(wind_direction_10m[j])
    hourly_data['当前短波太阳辐射-即时'] = tuple_remove(shortwave_radiation_instant[j])
    hourly_data['是否白天-0黑1白'] = tuple_remove(is_day[j])
    hourly_datas.append(hourly_data)

with open(r'C:\Users\Lst12\Desktop\时刻天气数据.csv', 'w', encoding='utf-8-sig', newline='') as f:
            headers = ['时刻', '当前温度-2m', '当前相对湿度-2m','当前降水量','当前体感温度','天气编码','当前风速-10m','当前总云量','当前风向','当前短波太阳辐射-即时','是否白天-0黑1白']
            writer = csv.DictWriter(f,headers)
            writer.writeheader()
            writer.writerows(hourly_datas)

