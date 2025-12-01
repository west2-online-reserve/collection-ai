import requests
import json
import time

API_LIST_URL = "https://summer-ospp.ac.cn/api/getProList"
API_DETAIL_URL = "https://summer-ospp.ac.cn/api/getProDetail"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": "https://summer-ospp.ac.cn",
    "Referer": "https://summer-ospp.ac.cn/"
}

def fetch_project_list():
    print(" 正在获取项目列表...")
    payload = {
        "supportLanguage": [], "techTag": [], "programmingLanguageTag": [],
        "programName": "", "difficulty": [], "lang": "zh", "orgName": [],
        "pageNum": 1, "pageSize": 50
    }
    try:
        res = requests.post(API_LIST_URL, headers=HEADERS, json=payload)
        return res.json().get('rows', [])
    except Exception as e:
        print(f" 列表请求失败: {e}")
        return []

def fetch_project_detail(program_code):
    """
    Fetch detailed information for a specific project from the Summer OSPP API.
    Args:
        program_code (str): The unique code identifying the project.
    Returns:
        dict: A dictionary containing the project's detailed information, or an empty dict if the request fails.
    """
    payload = {"programId": program_code, "type": "org"}
    try:
        res = requests.post(API_DETAIL_URL, headers=HEADERS, json=payload)
        return res.json()
    except Exception:
        return {}

def parse_requirements(req_data):
    if not req_data:
        return "暂无"

    # 如果已经是字符串，直接返回
    if isinstance(req_data, str):
        return req_data

    if isinstance(req_data, list):
        texts = []
        for item in req_data:
            if isinstance(item, dict) and 'title' in item:
                texts.append(item['title'])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)  # 用换行符拼接

    return str(req_data)

def main():
    projects = fetch_project_list()
    if not projects:
        print(" 列表为空")
        return

    final_results = []

    for index, proj in enumerate(projects):
        p_code = proj.get("programCode")
        p_name = proj.get("programName")

        print(f"正在抓取详情: {p_name} ")

        # 请求详情
        detail = fetch_project_detail(p_code)

        item = {
            "项目名称": p_name,
            "项目难度": detail.get("difficulty") or proj.get("difficulty"),
            "技术领域": detail.get("techTag"),  # 这是一个 JSON 字符串
            "项目简述": detail.get("programDesc") or "暂无",
            "产出要求": parse_requirements(detail.get("outputRequirement"))
        }

        final_results.append(item)
        time.sleep(0.2)

        # 保存
    with open("ospp_2025.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n数据已保存至 ospp_2025.json")

if __name__ == "__main__":
    main()