import requests
import json
import csv
def main():
    api = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 26.05942,
        "longitude": 119.198,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                  "sunshine_duration"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code",
                   "cloud_cover", "wind_speed_10m", "wind_direction_10m", "is_day", "shortwave_radiation_instant"],
        "timezone": "Asia/Shanghai"
    }
    response = requests.get(api, params=params)
    data = json.loads(response.content)
    d_writer = csv.writer(open('weather_daily.csv', 'w', encoding='utf8', newline=''))
    d_writer.writerow(['日期', '平均温度(°C)', '最高温度(°C)', '最低温度(°C)', '降水量(mm)', '日照时长(秒)'])
    for i in range(len(data['daily']['time'])):
        time = data['daily']['time'][i]
        temperature_2m_mean = data['daily']['temperature_2m_mean'][i]
        temperature_2m_max = data['daily']['temperature_2m_max'][i]
        temperature_2m_min = data['daily']['temperature_2m_min'][i]
        precipitation_sum = data['daily']['precipitation_sum'][i]
        sunshine_duration = data['daily']['sunshine_duration'][i]
        d_writer.writerow(
            [time, temperature_2m_mean, temperature_2m_max, temperature_2m_min, precipitation_sum, sunshine_duration])

    h_writer = csv.writer(open('weather_hourly.csv', 'w', encoding='utf8', newline=''))
    h_writer.writerow(['时间', '温度(°C)', '相对湿度(%)', '体感温度(°C)', '降水量(mm)', '天气代码(wmo code)', '云量(%)',
                       '10米风速(km/h)', '10米风向(°)', '白天/黑夜', '短波太阳辐射(W/m²)'])

    for i in range(len(data['hourly']['time'])):
        time = data['hourly']['time'][i]
        temperature_2m = data['hourly']['temperature_2m'][i]
        relative_humidity_2m = data['hourly']['relative_humidity_2m'][i]
        apparent_temperature = data['hourly']['apparent_temperature'][i]
        precipitation = data['hourly']['precipitation'][i]
        weather_code = data['hourly']['weather_code'][i]
        cloud_cover = data['hourly']['cloud_cover'][i]
        wind_speed_10m = data['hourly']['wind_speed_10m'][i]
        wind_direction_10m = data['hourly']['wind_direction_10m'][i]
        is_day = "白天" if data['hourly']['is_day'][i] == 1 else "黑夜"
        shortwave_radiation_instant = data['hourly']['shortwave_radiation_instant'][i]
        h_writer.writerow(
            [time, temperature_2m, relative_humidity_2m, apparent_temperature, precipitation, weather_code, cloud_cover,
             wind_speed_10m, wind_direction_10m, is_day, shortwave_radiation_instant])

if __name__ == '__main__':
    main()