import random
import requests
import csv
import time
from pprint import pprint
import json
from lxml import etree

def get_ospp():
    headers1 = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
    }
    response1_jsons = []   
    for page_num in range(1,13):
        datas1 = {  #要获得较多json串需要post的json参数，这仨个是必填参数，
            "pageNum": page_num,
            "pageSize": 50,
            "programName": "",
        }

        response1 = requests.post('https://summer-ospp.ac.cn/api/getProList', data = datas1, headers = headers1)
        response1.encoding = 'utf-8' 
        response1_json = json.loads(response1.text)['rows']  #解码json串并取rows对应的值，其他的键值对去掉
        for kv in response1_json:  #把获得的字典整合添加到一个新的字典response1_json中，存储所有项目信息
            response1_jsons.append(kv)
        time.sleep(random.uniform(2,5))

    projects_info = []
    for project in response1_jsons:  #遍历每个项目字典，寻找对应信息
        pr_id = project['programCode']  #获得id
        pr_difficulty = project['difficulty']   #获得难度
        pr_name = project['programName']  #获得项目名

        pr_domain_str = project['techTag'] #获得技术领域，原本的形式[[]]是字符串形式，需要json解析
        pr_domain_overall_list = json.loads(pr_domain_str) #再用json解析一次
        pr_domains = [pr_domains_list[1] for pr_domains_list in pr_domain_overall_list]  #技术领域有多个，需要成列表保存再分解出来
        
    

        headers2 = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
            "cookie":"tgt=1761145398.957.797.37771|2125b4d7ab6829ae59e8509cd0133f66; UM_distinctid=19a0c7223d5181c-05dc6c7c84342c-4c657b58-168000-19a0c7223d6182"
        }
        datas2 = {
            "programId": pr_id,
            type: 'org' 

        }   #项目简述和产出要求再另一个API，再post请求一次
        response2 = requests.post('https://summer-ospp.ac.cn/api/getProDetail', data = datas2)
        response2.encoding = 'utf-8'   #content type是text，需要解码工具
        response2_json = json.loads(response2.text) #获得全部的json串
        output_Requirement_list = response2_json["outputRequirement"]  #获得产出要求的总列表

        output_Requirement_parts = []
        output_Requirement_dicts =[output_Requirement for output_Requirement in output_Requirement_list[1:] if isinstance(output_Requirement , dict)]  #isinstance判断元素类型，筛选出所有字典类型元素
        for output_Requirement_dict in output_Requirement_dicts:   #遍历字典元素，把每个字典元素里的value一一筛选下来，并添加到一个新列表里，有两个key不好直接列表推导式，感觉顺序会乱
            output_Requirement_parts.append(output_Requirement_dict['title'])
            output_Requirement_parts.extend(output_Requirement_dict['children'])  #把作为可迭代的列表对象里的元素添加到另一个列表用extend
        
        pr_brief_html = response2_json["programDesc"]   #所需要的文本包含在html形式里，可以用etree解析
        pr_brief_text = etree.HTML(pr_brief_html)
        pr_brief = pr_brief_text.xpath('string(.)') #解析完用xpath的string(.)提取所有文本，记得加引号！！！！


        project_info = {

            '项目名': pr_name,
            '技术领域':','.join(pr_domains),
            '项目难度': pr_difficulty,
            '项目简述':pr_brief,
            '产出要求':','.join(output_Requirement_parts),

        }
        projects_info.append(project_info)
        #pprint(project_info,indent=4) 
        time.sleep(random.uniform(2,5))

    with open(r'C:\Users\Lst12\Desktop\开源之夏.csv', 'w', encoding='utf-8-sig', newline='') as f:
                headers = ['项目名', '技术领域', '项目难度','项目简述','产出要求']
                writer = csv.DictWriter(f,headers)
                writer.writeheader()
                writer.writerows(projects_info)

if __name__ == "__main__":
    get_ospp()

