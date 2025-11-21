import requests
import csv
import json

def get_weather_data_simple():
    session = requests.Session()
    session.trust_env = False
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    api = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 26.05942,
        "longitude": 119.198,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "sunshine_duration"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "weather_code",
                   "cloud_cover", "wind_speed_10m", "wind_direction_10m", "is_day", "shortwave_radiation_instant"],
        "timezone": "Asia/Shanghai"
    }
    
    try:
        response = requests.get(api, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 保存每日数据
        with open('weather_daily.csv', 'w', encoding='utf-8', newline='') as daily_file:
            d_writer = csv.writer(daily_file)
            d_writer.writerow(['Date', 'Mean Temperature (°C)', 'Max Temperature (°C)', 'Min Temperature (°C)', 
                              'Precipitation (mm)', 'Sunshine Duration (s)'])
            
            for i in range(len(data['daily']['time'])):
                time = data['daily']['time'][i]
                temperature_2m_mean = data['daily']['temperature_2m_mean'][i]
                temperature_2m_max = data['daily']['temperature_2m_max'][i]
                temperature_2m_min = data['daily']['temperature_2m_min'][i]
                precipitation_sum = data['daily']['precipitation_sum'][i]
                sunshine_duration = data['daily']['sunshine_duration'][i]
                d_writer.writerow([
                    time, 
                    temperature_2m_mean, 
                    temperature_2m_max, 
                    temperature_2m_min, 
                    precipitation_sum, 
                    sunshine_duration
                ])

        # 保存每小时数据
        with open('weather_hourly.csv', 'w', encoding='utf-8', newline='') as hourly_file:
            h_writer = csv.writer(hourly_file)
            h_writer.writerow(['Time', 'Temperature (°C)', 'Relative Humidity (%)', 'Apparent Temperature (°C)', 'Precipitation (mm)', 'Weather Code (WMO)', 
                'Cloud Cover (%)','Wind Speed 10m (km/h)', 'Wind Direction 10m (°)', 'Day/Night', 'Shortwave Radiation (W/m²)'
            ])

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
                is_day = "Day" if data['hourly']['is_day'][i] == 1 else "Night"
                shortwave_radiation_instant = data['hourly']['shortwave_radiation_instant'][i]
                h_writer.writerow([
                    time, temperature_2m, relative_humidity_2m,apparent_temperature, precipitation, weather_code,
                    cloud_cover,wind_speed_10m,wind_direction_10m,is_day,shortwave_radiation_instant
                ])
        
        print(f"Data saved successfully: {len(data['daily']['time'])} daily records, {len(data['hourly']['time'])} hourly records")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    get_weather_data_simple()