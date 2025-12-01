import requests
import json
import time
import csv
import re
import html as html_parser

all_projects = []
def get_projects():
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        }
    url='https://summer-ospp.ac.cn/api/getProList'
    for page_num in range(1, 13):  # 假设最多12页
        print(f"正在获取第 {page_num} 页...")
        payload = {
            "pageNum": str(page_num),
            "pageSize": "50",
            "programName": "",
            "supportLanguage": [],
            "techTag": [],
            "programmingLanguageTag": [],
            "difficulty": [],
            "lang": "zh",
            "orgName": []
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                datas = response.json()
                projects = datas.get('rows', [])
                for project in projects:
                    all_projects.append(project)
        except Exception as e:
            print(f"第 {page_num} 页请求异常: {e}")
            break

def parse_output_requirements(output_requirement_list):
    """解析产出要求列表"""
    if not output_requirement_list:
        return ""
    requirements = []
    for item in output_requirement_list:
        if item and isinstance(item, dict) and 'title' in item:
            requirements.append(item['title'])
    return "\n".join([f"• {req}" for req in requirements]) if requirements else "无产出要求"

def parse_html_description(html_text):
    if not html_text:
        return ""
    clean_text = re.sub('<[^<]+?>', '', html_text) # 移除HTML标签
    clean_text = html_parser.unescape(clean_text)    # 解码HTML实体
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()    # 清理多余空白
    return clean_text

def extract_project_details(projects):
    projects_info = []
    for i, project in enumerate(projects, 1):
        try:
            project_id = project['programCode']
            project_name = project['programName']
            project_difficulty = project['difficulty']
            # 技术领域
            tech_tag_string = project['techTag']
            tech_tag_list = json.loads(tech_tag_string)
            technology_domains = [tag_pair[1] for tag_pair in tech_tag_list]
            headers2 = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
                "cookie": "tgt=1761145398.957.797.37771|2125b4d7ab6829ae59e8509cd0133f66; UM_distinctid=19a0c7223d5181c-05dc6c7c84342c-4c657b58-168000-19a0c7223d6182"
            }
            data = {
                "programId": project_id,
                "type": 'org' 
            }
            response2 = requests.post('https://summer-ospp.ac.cn/api/getProDetail', json=data, headers=headers2)
            if response2.status_code == 200:
                detail_data = response2.json()
                program_desc_html = detail_data.get('programDesc', '')
                project_description = parse_html_description(program_desc_html)  
                # 解析产出要求
                output_requirement_list = detail_data.get("outputRequirement", [])
                output_requirements = parse_output_requirements(output_requirement_list)
            else:
                print(f"获取项目详情失败: {response2.status_code}")
                project_description = "（获取失败）"
                output_requirements = "（获取失败）"
            # 构建项目信息
            project_info = {
                '项目ID': project_id,
                '项目名称': project_name,
                '项目难度': project_difficulty,
                '技术领域': ", ".join(technology_domains),  # 转换为字符串
                '项目简述': project_description,
                '项目产出要求': output_requirements,
            }
            projects_info.append(project_info)   
            # 添加延迟避免请求过快
            time.sleep(0.5)
        except Exception as e:
            print(f"处理项目 {project.get('programName', '未知')} 时出错: {e}")
            continue
    return projects_info  # 返回结果列表

def save_to_csv(data, filename='summer_data.csv'):
    if not data:
        print("没有数据可保存到CSV")
        return
    fieldnames = [
        '项目ID', '项目名称', '项目难度', '技术领域', 
        '项目简述', '项目产出要求'
    ]
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for project in data:
                writer.writerow(project) 
        print(f"数据已保存到 {filename}")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def main():
    print("开始获取开源之夏项目数据...")
    # 获取项目列表
    get_projects()
    projects_info = extract_project_details(all_projects)
    if projects_info:
        # 保存到CSV文件
        save_to_csv(projects_info)
    else:
        print("未获取到任何项目数据")

if __name__ == "__main__":
    main()