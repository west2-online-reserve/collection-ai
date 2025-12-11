# -*- coding: utf-8 -*-
import requests
import time
from lxml import etree


# url = "https://summer-ospp.ac.cn/org/projectlist"
url = "https://summer-ospp.ac.cn/api/getProList"

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

        # 写入文件
        f.write(f"【项目{idx}】\n")
        f.write(f"项目名称：{project_name}\n")
        f.write(f"所属组织：{org_name}\n")
        f.write(f"难度：{difficulty}\n")
        f.write(f"技术标签：{tech_tag}\n")
        f.write(f"项目编码：{program_code}\n\n")

print("over!!数据已存储")

resp.close()
