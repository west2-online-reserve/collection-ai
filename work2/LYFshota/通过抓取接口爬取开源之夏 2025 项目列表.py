import requests
import csv
import json
import urllib.parse
import pathlib
from bs4 import BeautifulSoup
import re
import warnings
from requests.exceptions import SSLError
import ssl


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def flatten_output_requirement(data) -> str:
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()
    if not isinstance(data, list):
        return str(data).strip()
    parts: list[str] = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
        elif isinstance(item, dict):
            title = (item.get("title") or "").strip()
            if title:
                parts.append(title)
            children = item.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, str):
                        child_text = child.strip()
                        if child_text:
                            parts.append(child_text)
        # 其他类型忽略
    return " | ".join(parts)

if __name__ == "__main__":
    base_url = 'https://summer-ospp.ac.cn'
    headers = {
        'Host': 'summer-ospp.ac.cn',
        'Connection': 'keep-alive',
        'sec-ch-ua-platform': '"Windows"',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0',
        'sec-ch-ua': '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
        'Content-Type': 'application/json',
        'sec-ch-ua-mobile': '?0',
        'Accept': '*/*',
        'Origin': 'https://summer-ospp.ac.cn',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://summer-ospp.ac.cn/org/projectlist?lang=zh&pageNum=1&pageSize=50',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6'
    }
    cookies = {
        'tgt': '1764233499.812.797.219138|2125b4d7ab6829ae59e8509cd0133f66',
        'UM_distinctid': '19ac482df7b57b-0f2abd09aed7c2-4c657b58-1fa400-19ac482df7cd0b',
        'CNZZDATA1281243141': '1026344460-1764233503-https%253A%252F%252Fgithub.com%252F%7C1764428674'
    }
    ssl._create_default_https_context = ssl._create_unverified_context
    out_path = pathlib.Path(__file__).parent / "开源之夏2025项目数据"
    out_path.mkdir(exist_ok=True)
    csv_path = out_path / "开源之夏2025项目数据.csv"
    project_records = ['项目名称','项目难度','项目链接','技术领域标签','项目简述','项目产出要求']
    
    prolist_url = f'{base_url}/api/getProList'
    page_size = 50
    payload = {
        "keyword": "",
        "pageNum": 1,
        "pageSize": page_size,
        "lang": "zh",
        "trackName": "",
        "organizationName": "",
        "projectType": "",
        "scopeType": ""
    }

    project_rows: list[dict] = []
    for page_num in range(1, 13):
        payload["pageNum"] = page_num
        visible_url = f"{base_url}/org/projectlist?lang=zh&pageNum={page_num}&pageSize={page_size}"
        print(f"正在抓取列表页：{visible_url}")

        prolist_response = requests.post(
            prolist_url,
            headers=headers,
            cookies=cookies,
            timeout=10,
            json=payload,
            verify=False
        )

        prolist_response.encoding = 'utf-8'
        if not prolist_response.ok:
            raise RuntimeError(
                f"列表页请求失败（page={page_num}）：{prolist_response.status_code} {prolist_response.reason}\n"
                f"{prolist_response.text[:500]}"
            )

        try:
            prolist_data = prolist_response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"项目列表接口返回的并非有效 JSON（page={page_num}），无法解析"
            ) from exc

        current_rows = (
            prolist_data.get("rows")
            or prolist_data.get("data")
            or prolist_data.get("list")
            or []
        )
        if not current_rows:
            print(f"第 {page_num} 页无项目数据，提前结束翻页")
            break

        project_rows.extend(current_rows)
        print(f"第 {page_num} 页获取 {len(current_rows)} 条，累计 {len(project_rows)} 条")

    if not project_rows:
        warnings.warn("未获取到任何项目记录，请检查接口参数")

    project_records = [project_records]  # 将表头转换为二维列表的第一行

    for project in project_rows:
        if not isinstance(project, dict):
            continue
        project_name = (project.get("programName") or "").strip()
        project_difficulty = (project.get("difficulty") or "").strip()
 
        program_code = (project.get("programCode") or "").strip()
        project_link = (
            f"{base_url}/org/projectdetail?programCode={urllib.parse.quote(program_code)}"
            if program_code
            else ""
        )

        project_records.append([
            project_name,
            project_difficulty,
            project_link,
            "",
            "",
            "",
        ])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(project_records)

    print(f"已保存开源之夏2025项目数据到 {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)
    detail_headers_template = {
        'Host': 'summer-ospp.ac.cn',
        'Connection': 'keep-alive',
        'Content-Length': '38',
        'sec-ch-ua-platform': '"Windows"',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0',
        'sec-ch-ua': '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
        'Content-Type': 'application/json',
        'sec-ch-ua-mobile': '?0',
        'Accept': '*/*',
        'Origin': 'https://summer-ospp.ac.cn',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': '',  # 将在循环中按序号替换
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6'
    }
    detail_cookies = {
        'tgt': '1764233499.812.797.219138|2125b4d7ab6829ae59e8509cd0133f66',
        'UM_distinctid': '19ac482df7b57b-0f2abd09aed7c2-4c657b58-1fa400-19ac482df7cd0b',
        'CNZZDATA1281243141': '1026344460-1764233503-https%253A%252F%252Fgithub.com%252F%7C1764433721'
    }

    for idx, record in enumerate(records, start=1):
        project_link = record['项目链接']
        if not project_link:
            warnings.warn(f"项目 {record['项目名称']} 缺少项目链接，跳过详情抓取")
            continue
        match = re.search(r'programCode=([^&]+)', project_link)
        if not match:
            warnings.warn(f"无法从链接中提取 programCode：{project_link}")
            continue
        project_id = urllib.parse.unquote(match.group(1))
        print(f"正在处理项目：{record['项目名称']} -> {project_link}")
        project_url= f"{base_url}/api/getProDetail"
        request_headers = detail_headers_template.copy()
        request_headers['Referer'] = f'https://summer-ospp.ac.cn/org/prodetail/{idx}?lang=zh&list=pro'


        project_response = requests.post(
            project_url,
            headers=request_headers,
            cookies=detail_cookies,
            timeout=10,
            json={"programId": project_id, "type": "org", "lang": "zh"},
            verify=False
        )
        project_response.encoding = 'utf-8'
        if not project_response.ok:
            warnings.warn(
                f"项目 {record['项目名称']} 详情请求失败：{project_response.status_code} {project_response.reason}\n"
                f"响应片段：{project_response.text[:200]}"
            )
            continue
        try:
            project_data = project_response.json()
        except json.JSONDecodeError as exc:
            preview = project_response.text[:200]
            warnings.warn(
                f"项目 {record['项目名称']} 详情返回的并非 JSON，跳过。响应片段：{preview}"
            )
            continue
        project_description = html_to_text(project_data.get("programDesc") or "")
        project_output = flatten_output_requirement(project_data.get("outputRequirement"))
        record['项目简述'] = project_description
        record['项目产出要求'] = project_output
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = ['项目名称','项目难度','项目链接','技术领域标签','项目简述','项目产出要求']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


    
    
    