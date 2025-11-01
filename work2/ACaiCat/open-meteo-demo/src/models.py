from __future__ import annotations

from typing import List

from pydantic import BaseModel


class HourlyUnits(BaseModel):
    time: str
    temperature_2m: str
    relative_humidity_2m: str
    apparent_temperature: str
    precipitation: str
    weather_code: str
    cloud_cover: str
    wind_speed_10m: str
    wind_direction_10m: str
    is_day: str
    shortwave_radiation: str


class Hourlies(BaseModel):
    time: List[str]
    temperature_2m: List[float]
    relative_humidity_2m: List[int]
    apparent_temperature: List[float]
    precipitation: List[float]
    weather_code: List[int]
    cloud_cover: List[int]
    wind_speed_10m: List[float]
    wind_direction_10m: List[int]
    is_day: List[int]
    shortwave_radiation: List[float]


class Hourly(BaseModel):
    time: str
    temperature_2m: float
    relative_humidity_2m: int
    apparent_temperature: float
    precipitation: float
    weather_code: int
    cloud_cover: int
    wind_speed_10m: float
    wind_direction_10m: int
    is_day: int
    shortwave_radiation: float


class DailyUnits(BaseModel):
    time: str
    precipitation_sum: str
    sunshine_duration: str
    temperature_2m_min: str
    temperature_2m_max: str
    temperature_2m_mean: str


class Dailies(BaseModel):
    time: List[str]
    precipitation_sum: List[float]
    sunshine_duration: List[float]
    temperature_2m_min: List[float]
    temperature_2m_max: List[float]
    temperature_2m_mean: List[float]


class Daily(BaseModel):
    time: str
    precipitation_sum: float
    sunshine_duration: float
    temperature_2m_min: float
    temperature_2m_max: float
    temperature_2m_mean: float


class APIResponse(BaseModel):
    latitude: float
    longitude: float
    generationtime_ms: float
    utc_offset_seconds: int
    timezone: str
    timezone_abbreviation: str
    elevation: float
    hourly_units: HourlyUnits
    hourly: Hourlies
    daily_units: DailyUnits
    daily: Dailies
