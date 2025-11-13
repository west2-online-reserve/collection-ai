import requests
import json
import csv
import pandas as pd

def out_list(list,message_name):
    new_dict = {}
    for i in list:
        if isinstance(i,list):
            out_list(i,message_name)
        elif isinstance(i,dict):
            out_list(i,message_name)
        elif i in message_name:
            new_dict[i] = list[i]


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 26.05942,
	"longitude": 119.198,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum", "sunshine_duration"],
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"timezone": "Asia/Singapore",
}
def get_message(url,params):
    web = requests.get(url,params=params)
    print('1')
    text =web.text
    data =web.json()
    with open('data.json','w',encoding='utf-8-sig') as f:
        json.dump(data,f)

with open("data.json",'r',encoding='utf-8-sig') as f:
    data = json.load(f)

df1 = pd.DataFrame(data["hourly"])
df1.to_csv('data.csv',mode='a',index=False)
df2 = pd.DataFrame(data["daily"])
df2.to_csv('data.csv',mode='a',index=False)

hourly_units = data["hourly_units"]
hourly_units_list = []
for i in hourly_units:
    hourly_units_list.append(f"{i}({hourly_units[i]})")

print(hourly_units_list)

'''
with open('data.csv','w',encoding='utf-8-sig') as f:
    writer = csv.Writer(f)
    writer.writerows(data)
'''
