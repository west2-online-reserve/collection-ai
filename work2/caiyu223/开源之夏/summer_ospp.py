import requests
import json
import time
import csv


def get_list(page):

    body = {
        "difficulty": [],
        "lang": "zh",
        "orgName": [],
        "pageNum": "1", 
        "pageSize": "50",
        "programName": "",
        "programmingLanguageTag": [],
        "supportLanguage": [],
        "techTag": []
    }
    headers ={
    'Referer': f'https://summer-ospp.ac.cn/org/projectlist?lang=zh&pageNum={page}&pageSize=50&programName='
    }

    list_url='https://summer-ospp.ac.cn/api/getProList'
    web1=requests.post(list_url,json=body,headers=headers)
    list_data=web1.json()
    programCodes=[i["programCode"] for i in list_data["rows"]]
    return programCodes
    #print(programCodes)

def get_detail(prograncode):
    time.sleep(0.5)
    text_body={'programId': f"{prograncode}", 'type': "org"}
    url ='https://summer-ospp.ac.cn/api/getProDetail'

    web =requests.post(url,json=text_body)
    data = web.json()
    print(json.dumps(data,indent=2, ensure_ascii=False))
    main_data={
        '项目名':f"{data['outputRequirement']}",
        '项目难度':f"{data['difficulty']}",
        '技术领域标签':f'{data["techTag"]}',
        '项目简述':f'{data["programDesc"]}',
        '项目产出要求':f'{data["techRequirement"]}'
    }
    #print(main_data)
    with open('program_detail.json','a',encoding='utf-8-sig') as f:
        json.dump(main_data,f,ensure_ascii=False)
    with open('program_detail.csv','a',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=fieldname)
        writer.writerow(main_data)

    #print(web.text)

fieldname = ['项目名','项目难度','技术领域标签','项目简述','项目产出要求']
with open('program_detail.csv','w',encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f,fieldnames=fieldname)
    writer.writeheader()
page = 1
final_page = 2
get_program_num = 10
while page<=final_page:
    page += 1
    programcodes = get_list(page)
    print(programcodes)
    program_num = 1
    for i in programcodes:
        if program_num>get_program_num:
            break
        program_num += 1
        get_detail(i)

