# -*- coding: utf-8 -*-
import requests
import time
from lxml import etree
import re


def clean_html(html_str):
    if not html_str:
        return "无项目简述"
    # 移除所有HTML标签
    clean_text = re.sub(r'<[^>]+>', '', html_str)
    # 合并多余的换行/空格，保持格式整洁
    clean_text = re.sub(r'\n+', '\n', clean_text).strip()
    return clean_text


# url = "https://summer-ospp.ac.cn/org/projectlist"
url = "https://summer-ospp.ac.cn/api/getProList"
url_message = "https://summer-ospp.ac.cn/api/getProDetail"
sub_url = "https://summer-ospp.ac.cn/org/prodetail/"
sub_url_1 = "?lang=zh&list=pro"


page = int(input("你想爬取第几页的数据："))


data = {
    "difficulty": [],
    "lang": "zh",
    "orgName": [],
    "pageNum": page,
    "pageSize": 50,
    "programName": "",
    "programmingLanguageTag": [],
    "supportLanguage": [],
    "techTag": []
}

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
    "referer": "https://summer-ospp.ac.cn/org/projectlist?lang=zh&pageNum=1&pageSize=50&programName=",
    "cookie": "tgt=1765425913.542.797.222039|2125b4d7ab6829ae59e8509cd0133f66",
    "origin": "https://summer-ospp.ac.cn",
    "content-type": "application/json;charset=UTF-8"
}

resp = requests.post(url, headers=headers, json=data)

# print(resp.text)
data_json = resp.json()

if data_json.get("code") != 200:
    print("接口请求失败：", data_json.get("msg"))
    exit()

total_projects = data_json.get("total", 0)
project_rows = data_json.get("rows", [])
print(f"当前页项目数量：{len(project_rows)}\n")

# 打开文件并写入
with open("开源之夏项目列表.txt", "w", encoding="utf-8") as f:
    f.write(f"接口返回总项目数：{total_projects}\n")
    f.write(f"当前页项目数量：{len(project_rows)}\n")
    f.write(f"当前页数：{page}\n\n")

    for idx, project in enumerate(project_rows, 1):
        project_name = project.get("programName", "未知项目名")
        org_name = project.get("orgName", "未知组织")
        difficulty = project.get("difficulty", "未知难度")
        tech_tag = project.get("techTag", "无技术标签")
        program_code = project.get("programCode", "无编码")

        # 请求子页面所需的参数
        data_message = {
            "programId": program_code,
            "type": "org"
        }
        # 子页面信息的请求
        resp_message = requests.post(url_message, headers=headers, json=data_message)
        message_json = resp_message.json()

        # 提取并清理项目简述（含项目背景/任务）
        program_desc_html = message_json.get("programDesc", "")
        program_desc = clean_html(program_desc_html)

        # 解析项目产出要求
        project_requirements = message_json.get("outputRequirement", [])
        requirements_text = ""  # 初始化总要求文本

        for project_requirement in project_requirements:
            # 跳过None元素
            if project_requirement is None:
                continue

            # 提取产出要求的标题（当前接口中children为空，只需取title）
            req_title = project_requirement.get("title", "")
            if req_title:
                requirements_text += f"- {req_title}\n"

        f.write(f"【项目{idx}】\n")
        f.write(f"项目名称：{project_name}\n")
        f.write(f"所属组织：{org_name}\n")
        f.write(f"难度：{difficulty}\n")
        f.write(f"技术标签：{tech_tag}\n")
        f.write(f"项目编码：{program_code}\n")
        f.write(f"项目简述：\n{program_desc}\n")  # 新增项目简述
        f.write(f"项目产出要求：\n{requirements_text if requirements_text else '无'}\n\n")

print("over!!数据已存储")

resp.close()
