import requests



headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"}
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 26.05942,
	"longitude": 119.198,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "sunshine_duration", "precipitation_sum"],
	"hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation_instant", "is_day"],
}

responses = requests.get(url=url,params=params,headers=headers).json()
# 写入CSV文件
with open('task4.csv', 'w', encoding='utf-8-sig') as f:
    # 写入表头
    f.write('日期,时间,平均温度(2 米),最高温度(2 米),最低温度(2 米),日照时长,降水量,温度(2 米),相对湿度(2 米),体感温度,降水(雨 + 雪),天气代码,总云量,风速(10 米),风向(10 米),短波太阳辐射 GHI(即时),是否白天\n')
    
    # 遍历数据
    for r in range(len(responses["hourly"]["time"])):
        h = responses["hourly"]
        d = responses["daily"]
        formatted_date = d["time"][r//24]
        formatted_time = h["time"][r][-5:]  # 提取最后5位作为时间
        is_day = "Yes" if h["is_day"][r] else "No"
        
        if r % 24 == 0:
            day_index = r // 24
            f.write(f'{formatted_date},{formatted_time},')
            f.write(f'{d["temperature_2m_mean"][day_index]},')
            f.write(f'{d["temperature_2m_max"][day_index]},')
            f.write(f'{d["temperature_2m_min"][day_index]},')
            f.write(f'{d["sunshine_duration"][day_index]},')
            f.write(f'{d["precipitation_sum"][day_index]},')
        else:
            f.write(f',{formatted_time},,,,,,')  # 日期留空，日统计数据留空
        
        # 写入小时数据（两种情况都需要）
        f.write(f'{h["temperature_2m"][r]},')
        f.write(f'{h["relative_humidity_2m"][r]},')
        f.write(f'{h["apparent_temperature"][r]},')
        f.write(f'{h["precipitation"][r]},')
        f.write(f'{h["weather_code"][r]},')
        f.write(f'{h["cloud_cover"][r]},')
        f.write(f'{h["wind_speed_10m"][r]},')
        f.write(f'{h["wind_direction_10m"][r]},')
        f.write(f'{h["shortwave_radiation_instant"][r]},')
        f.write(f'{is_day}\n')

print(f"数据写入完成，共 {len(responses['hourly']['time'])} 行")