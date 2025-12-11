import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


# - 通过浏览器抓包工具找到开源之夏项目列表的接口，并使用 requests 等库进行请求，获取到项目列表。
# - 获得每个项目的项目名、项目难度、技术领域标签。
# - 获取每个项目的具体信息，包括项目简述、项目产出要求。

program_num='1000'

api=r'https://summer-ospp.ac.cn/api/getProList'
payload ={
    "supportLanguage": [],
    "techTag": [],
    "programmingLanguageTag": [],
    "programName": "",
    "difficulty": [],
    "pageNum": "1",
    "pageSize": program_num,
    "lang": "zh",
    "orgName": []
}

print("获取项目列表")
response = requests.post(api, json=payload)

program_list=response.json()['rows']
program_codes=[program['programCode'] for program in program_list]

def fetch_program_datas(code):
    api=r'https://summer-ospp.ac.cn/api/getProDetail'
    payload={
    "programId": str(code),
    "type": "org"
    }
    r=requests.post(api,json=payload)
    js=r.json()

    name=js['programName']
    dif=js['difficulty']
    tag=js['techTag']

    soup=BeautifulSoup(js['programDesc'],'lxml')
    output_req=soup.text

    detail=''
    for content in js['outputRequirement']:
        if content and content['title']:
            detail+=content['title']
            detail+='\n'
    
    return {
        '项目名':name,
        '项目难度':dif,
        '技术领域标签':tag,
        '项目简述':detail,
        '项目产出要求':output_req,
    }

results=[]

with ThreadPoolExecutor(max_workers=50) as executor:
    future_to_path ={
        executor.submit(fetch_program_datas,code): code for code in program_codes
    }

    for future in as_completed(future_to_path):
        news_path=future_to_path[future]
        data=future.result()
        results.append(data)
        print(f"进度{len(results)}/{len(program_codes)}")

df=pd.DataFrame(results)
print(df)
df.to_csv("data.csv",index=False)


