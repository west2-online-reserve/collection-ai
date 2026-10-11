from typing import Any
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import requests
import os
import re
import json


file_path = os.path.dirname(os.path.abspath(__file__))
storage_file_path = os.path.join(file_path, '开源之夏项目数据.txt')
storage_folder_path = os.path.join(file_path, '开源之夏附件')
os.makedirs(storage_folder_path, exist_ok=True)


headers: dict[str, str] = {
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'content-type': 'application/json',
    'origin': 'https://summer.ospp.ac.cn',
    'priority': 'u=1, i',
    'referer': 'https://summer.ospp.ac.cn/org/projectlist?lang=zh&pageNum=1&pageSize=50',
    'sec-ch-ua': '"Chromium";v="152", "Not?A_Brand";v="24", "Microsoft Edge";v="152"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36 Edg/152.0.0.0',
}

list_url: str = 'https://summer.ospp.ac.cn/api/getProList'
detail_url: str = 'https://summer.ospp.ac.cn/api/getProDetail'
pdf_binary_url: str = "https://summer.ospp.ac.cn/api/publicApplication"

# 附件下载
def download_pdf(file_name: str, proId: str) -> None:
    save_path = os.path.join(storage_folder_path, file_name)
    payload = {"proId": proId}
    resp = requests.post(pdf_binary_url, headers=headers, json=payload, stream=True, timeout=20)
    resp.raise_for_status()

    if len(resp.content) >= 100:
        if resp.content.startswith(b"%PDF"):
            with open(save_path,"wb") as f:
                f.write(resp.content)
                print(f"已保存: {save_path}")

# 抓取标签
def tag_extract(raw_tag_text:str) -> list[str]:
    pattern = re.compile(r'\"(.*?)\".*?')
    return re.findall(pattern, raw_tag_text)

# html 转化
def html_to_plain_text(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "html.parser")
    text: str = soup.get_text(separator="\n", strip=True)
    return text

# 整合列表
def list_extract(raw_list: list[Any]) -> list[str]:
    new_list: list[str] = []
    for x in raw_list:
        if x != None:
            if x['title']:
                new_list.append(x['title'])
    return new_list

# 从详细页的 response 源码中提取信息
def extractor(raw_json_text: dict[str, Any]) -> dict[str, str | list[str]]:
    brief_text: str = html_to_plain_text(raw_json_text["programDesc"])
    output_requirement_list:list[str] = list_extract(raw_json_text["outputRequirement"])
    tech_requirement_list:list[str] = list_extract(raw_json_text["techRequirement"])
    return {"项目简介":brief_text, "产出要求":output_requirement_list, "技术要求":tech_requirement_list}

# 获取详细页的请求数据
def get_data(code: str) -> dict[str, str]:
    return {'programId': code, 'type': 'org'}

# 获取详细页的信息
def get_detail(raw_program: dict[str, Any]) -> dict[str, str | list[str]]:
    detail_grams: dict[str, str] = get_data(raw_program['programCode'])
    response = requests.post(detail_url, headers=headers, json=detail_grams)
    text: dict[str, Any] = json.loads(response.text)
    print(f"{raw_program['programCode']}：处理完毕")
    return extractor(text)

# 获取一页的信息，并下载附件
def get_one_page_information(page_num:int) -> list[dict[str, Any]]:
    json_data = {
        'supportLanguage': [],
        'techTag': [],
        'programmingLanguageTag': [],
        'programName': '',
        'difficulty': [],
        'completionTime': [],
        'pageNum': '1',
        'pageSize': '50',
        'programType': page_num,
        'lang': 'zh',
        'orgName': [],
    }
    response = requests.post(list_url, headers=headers, json=json_data)
    raw_data: dict[str, Any] = json.loads(response.text)
    rows: list[dict[str, Any]] = raw_data['rows']
    data_list: list[dict[str, Any]] = []
    for raw_program in rows:
        rp: dict[str, Any] = raw_program
        detail_dict: dict[str, str | list[str]] = get_detail(rp)
        tag = tag_extract(rp['techTag'])
        program: dict[str, Any] = {"项目名":rp['programName'], "难度":rp['difficulty'], "标签":tag} | detail_dict
        data_list.append(program)

        accessory_name: str = rp['programCode'] + '.pdf'
        download_pdf(accessory_name, rp['proId'])
    return data_list


page_1_list: list[dict[str, Any]] = get_one_page_information(1)
page_2_list: list[dict[str, Any]] = get_one_page_information(2)

all_list = page_1_list + page_2_list

with open(storage_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_list, f, ensure_ascii=False, indent=2)
print("输出存储完成！")