from fake_useragent import UserAgent

BASE_URL: str = "https://jwch.fzu.edu.cn/"

HEADERS: dict[str, str] = {
    "user-agent": UserAgent().random
}