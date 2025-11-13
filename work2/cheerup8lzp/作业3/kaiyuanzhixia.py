import requests
import json
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def get_chrome_driver():
    """创建并返回配置好的Chrome WebDriver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--log-level=3')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    return webdriver.Chrome(options=options)

def fetch_project_details(project_id):
    """获取项目详情页的简述和产出要求"""
    detail_url = f"https://summer-ospp.ac.cn/org/prodetail/{project_id}?lang=zh&list=pro"
    driver = None
    
    try:
        driver = get_chrome_driver()
        driver.get(detail_url)
        time.sleep(3)
        
        # 获取项目简述
        try:
            project_summary = driver.find_element(By.CSS_SELECTOR, ".bf-content").text.strip()
        except:
            project_summary = "暂无简述"
        
        # 获取项目产出要求
        try:
            panes = driver.find_elements(By.CSS_SELECTOR, ".pane")
            project_requirements = panes[0].text.strip() if panes else "暂无产出要求"
        except:
            project_requirements = "暂无产出要求"
        
        return project_summary, project_requirements
        
    except Exception as e:
        print(f"获取项目 {project_id} 详情失败: {e}")
        return "暂无简述", "暂无产出要求"
    finally:
        if driver:
            driver.quit()

def fetch_project_list():
    """爬取项目列表并保存到CSV文件"""
    api_url = "https://summer-ospp.ac.cn/api/getProList"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {"lang": "zh", "pageNum": 1, "pageSize": 50}
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        projects = response.json().get("rows", [])
        
        # 存储所有项目数据
        project_data = []
        
        for index, project in enumerate(projects, 1):
            project_name = project.get("programName", "未知项目名")
            project_id = project.get("programCode", "未知项目编号")
            project_difficulty = project.get("difficulty", "未知难度")
            
            # 解析技术标签
            try:
                tech_tags = json.loads(project.get("techTag", "[]"))
                tech_tag_list = ", ".join([tag[1] for tag in tech_tags if len(tag) > 1])
            except:
                tech_tag_list = ""
            
            print(f"正在爬取第 {index}/{len(projects)} 个项目: {project_name}")
            
            # 获取详情页信息
            project_summary, project_requirements = fetch_project_details(project_id)
            
            # 添加到数据列表
            project_data.append({
                "项目名": project_name,
                "项目难度": project_difficulty,
                "技术领域标签": tech_tag_list,
                "项目简述": project_summary,
                "项目产出要求": project_requirements
            })
        
        # 写入CSV文件
        with open("projects.csv", "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["项目名", "项目难度", "技术领域标签", "项目简述", "项目产出要求"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(project_data)
        
        print(f"\n爬取完成！共爬取 {len(project_data)} 个项目，数据已保存到 projects.csv 文件")
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

# 运行爬虫
if __name__ == "__main__":
    fetch_project_list()