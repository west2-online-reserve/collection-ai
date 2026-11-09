import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.edge.options import Options

if __name__ == "__main__":
    options = Options()
    prefs = {
        "download.default_directory": str(Path(__file__).resolve().parent),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Edge(options=options)

    weahter_info_config = {
        "longitude": 119.198,
        "latitude": 26.05942,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "timezone": r"Asia%2FSingapore",
        "format": "csv",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "cloud_cover",
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
    }

    url = "https://archive-api.open-meteo.com/v1/archive?"

    for key, value in weahter_info_config.items():
        if not isinstance(value, list):
            url += f"{key}={value}"
        else:
            url += f"{key}="
            for var in value:
                url += f"{var},"

        url += "&"

    driver.get(url)
    time.sleep(10)

    driver.quit()
