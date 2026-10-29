import requests,os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#参数
latitude_ = '26.05942'
longtitude_ = '119.198'
start_data_ = '2024-01-01'
end_data_ = '2024-12-31'
daily_ = ('temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration')
hourly_ = ('temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation_instant,is_day')
param = {
'latitude':latitude_,
'longitude':longtitude_,
'start_date':start_data_,
'end_date':end_data_,
'daily':daily_,
'hourly':hourly_,
'format':'csv'
}
head = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0'

}
url = 'https://archive-api.open-meteo.com/v1/archive'
resp = requests.get(url,params=param,headers=head)
if resp.status_code == 200:
    with open('旗山校区天气.csv','wb') as f:
        f.write(resp.content)
    print('成功下载')
else:
    print(f"请求失败，请求码：{resp.status_code}")