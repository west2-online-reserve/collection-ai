import csv
import requests
from typing import Dict, List

# ------------------- 参数 -------------------
LATITUDE = 26.05942
LONGITUDE = 119.198
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
TIMEZONE   = "Asia/Shanghai"

HOURLY_VARS = [
    "temperature_2m",
    "relativehumidity_2m",
    "apparent_temperature",
    "precipitation",
    "weathercode",
    "cloudcover",
    "windspeed_10m",
    "winddirection_10m",
    "shortwave_radiation",
    "is_day",
]

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "sunshine_duration",
]

OUT_CSV = "fuzhou_qishan_2024_hourly_with_daily.csv"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URLS = [
    "https://archive.open-meteo.com/v1/archive",
    "https://archive-api.open-meteo.com/v1/archive",
]

# ------------------- 1. 请求 -------------------
def request_once() -> Dict:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
        "hourly": ",".join(HOURLY_VARS),
        "daily": ",".join(DAILY_VARS),
    }

    try:
        r = requests.get(FORECAST_URL, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        # 若 forecast 超出可用范围，改用 archive 接口
        for url in ARCHIVE_URLS:
            try:
                r = requests.get(url, params=params, timeout=120)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                continue
        raise  # 所有尝试都失败，向上抛异常

# ------------------- 2. 构建日数据映射 -------------------
def build_daily_map(daily: Dict) -> Dict[str, Dict[str, object]]:
    daily_map = {}
    dates = daily.get("time", [])
    for i, d in enumerate(dates):
        daily_map[d] = {var: daily.get(var, [None])[i] for var in DAILY_VARS}
    return daily_map

# ------------------- 3. 保存 CSV -------------------
def save_csv(hourly: Dict, daily_map: Dict[str, Dict[str, object]]) -> None:
    times = hourly.get("time", [])
    fieldnames = ["time"] + HOURLY_VARS + DAILY_VARS

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, ts in enumerate(times):
            row = {"time": ts}
            # hourly 列
            for var in HOURLY_VARS:
                row[var] = hourly.get(var, [None])[i]
            # 对应的日列（通过日期部分匹配）
            day = ts.split("T")[0]
            for var in DAILY_VARS:
                row[var] = daily_map.get(day, {}).get(var, None)
            writer.writerow(row)

    print(f"CSV 已写入 {OUT_CSV}")

# ------------------- 主流程 -------------------
def main() -> None:
    data = request_once()
    hourly = data.get("hourly", {})
    daily  = data.get("daily", {})

    daily_map = build_daily_map(daily)
    save_csv(hourly, daily_map)

if __name__ == "__main__":
    main()
