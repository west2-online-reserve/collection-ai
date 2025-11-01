import pandas
import requests

from models import APIResponse, Daily, Hourly


def main():
    params = {
        "latitude": 26.05942,
        "longitude": 119.198,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "daily": "precipitation_sum,sunshine_duration,temperature_2m_min,temperature_2m_max,temperature_2m_mean",
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,is_day,shortwave_radiation"
    }

    response: requests.Response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive", params=params)

    api_response: APIResponse = APIResponse.model_validate(response.json())
    daily_weathers: list[Daily] = []
    hourly_weathers: list[Hourly] = []
    for i in range(len(api_response.daily.time)):
        daily = api_response.daily
        daily_weathers.append(Daily(time=daily.time[i],
                                    precipitation_sum=daily.precipitation_sum[i],
                                    sunshine_duration=daily.sunshine_duration[i],
                                    temperature_2m_max=daily.temperature_2m_max[i],
                                    temperature_2m_min=daily.temperature_2m_min[i],
                                    temperature_2m_mean=daily.temperature_2m_mean[i]))
    for i in range(len(api_response.hourly.time)):
        hourly = api_response.hourly
        hourly_weathers.append(Hourly(time=hourly.time[i],
                                      temperature_2m=hourly.temperature_2m[i],
                                      relative_humidity_2m=hourly.relative_humidity_2m[i],
                                      apparent_temperature=hourly.apparent_temperature[i],
                                      weather_code=hourly.weather_code[i],
                                      precipitation=hourly.precipitation[i],
                                      cloud_cover=hourly.cloud_cover[i],
                                      wind_speed_10m=hourly.wind_speed_10m[i],
                                      wind_direction_10m=hourly.wind_direction_10m[i],
                                      is_day=hourly.is_day[i],
                                      shortwave_radiation=hourly.shortwave_radiation[i]))
    df_daily = pandas.DataFrame([i.model_dump() for i in daily_weathers])
    df_hourly = pandas.DataFrame([i.model_dump() for i in hourly_weathers])

    df_daily.to_csv("df_daily.csv", index=False)
    df_hourly.to_csv("df_hourly.csv", index=False)

    print(f"Export {len(df_daily)} daily weather records...")
    print(f"Export {len(df_hourly)} hourly weather records...")


if __name__ == "__main__":
    main()
