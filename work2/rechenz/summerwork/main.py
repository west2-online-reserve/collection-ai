import requests
import csv
from bs4 import BeautifulSoup
import json
with open('AIsolution/work2/summerwork/summer.csv', 'w', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['项目名', '项目难度', '技术领域标签', '项目简述', '项目产出要求'])
    num = 1
    for page in range(1, 13):
        listurl = 'https://summer-ospp.ac.cn/api/getProList'
        listrespones = requests.post(listurl, data={
                                     'pageNum': page, 'pageSize': 50, 'programName': ''})
        listrespones.encoding = 'utf-8'
        listrespones = json.loads(listrespones.text)
        # print(listrespones)
        it = iter(listrespones['rows'])
        for curprogramme in it:
            anslist = []
            anslist.append(curprogramme['programName'])
            anslist.append(curprogramme['difficulty'])
            anslist.append(curprogramme['techTag'])
            Orgdetailurl = 'https://summer-ospp.ac.cn/api/getOrgDetail'
            orgdetailresponse = requests.post(
                Orgdetailurl, data={'orgId': curprogramme['orgId']})
            orgdetailresponse.encoding = 'utf-8'
            orgdetailresponse = json.loads(orgdetailresponse.text)
            anslist.append(orgdetailresponse['full_des'])
            Prodetailurl = 'https://summer-ospp.ac.cn/api/getProDetail'
            prodetailresponse = requests.post(
                Prodetailurl, data={'programId': f"{curprogramme['programCode']}", type: "org"})
            prodetailresponse.encoding = 'utf-8'
            prodetailresponse = json.loads(prodetailresponse.text)
            itt = iter(prodetailresponse['outputRequirement'])
            flag = False
            for curoutput in itt:
                if flag == False:
                    flag = True
                    continue
                anslist.append(curoutput['title'])
            writer.writerow(anslist)
            print(f'项目{num}完成')
            num += 1
