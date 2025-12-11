import time
from lxml import etree
import requests
import os
import  csv
import re

base_url=r"https://summer-ospp.ac.cn/org/projectlist"
request_url=r"https://summer-ospp.ac.cn/api/getProList"
detail_url=r"https://summer-ospp.ac.cn/api/getProDetail"

data_list=[] #存储最终信息（除 pdf外）
dir_1 = "项目申请书"
if not os.path.exists(dir_1):
    os.mkdir(dir_1)

headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
}
data={ # 从'getProList'里找到的负载
    "supportLanguage":[],"techTag":[],"programmingLanguageTag":[],
    "programName":"","difficulty":[],"pageNum":"1",
    "pageSize":"50","lang":"zh","orgName":[]
}


response=requests.post(url=request_url,headers=headers,data=data)
json_text=response.json()
row_list=json_text["rows"]
num=0
print(f"共爬取到{len(row_list)}条项目信息：")
for row in row_list:
    pro_name=row["programName"] #项目名字
    pro_difficulty=row["difficulty"] #项目难度
    pro_techTag=row["techTag"] #项目技术标签
    num+=1
    print(f'正在处理第{num}个项目:"{pro_name}"')

    pro_Code=row["programCode"]
    detail_url="https://summer-ospp.ac.cn/api/getProDetail"
    data_detail = {  # 后续每个项目的详情里getProDetail的负载
        "programId": pro_Code, "type": "org"
    }
    time.sleep(1)
    response_detail=requests.post(url=detail_url,headers=headers,data=data_detail)
    json_detail_text=response_detail.json()


    pro_Desc_html=json_detail_text["programDesc"]
    pro_tree=etree.HTML(pro_Desc_html)
    pro_detail_list=pro_tree.xpath('//p//text()')
    pro_detail=' '.join(pro_detail_list) #项目详情


    pro_oReq_list=json_detail_text["outputRequirement"]
    pro_oReq_text_list=[] #项目产出要求
    for pro_oReq in pro_oReq_list:
        if pro_oReq:
            part=pro_oReq.get("title")
            pro_oReq_text_list.append(part)
    pro_Data={
        'name':pro_name,
        "diffculty":pro_difficulty,
        "techTag":pro_techTag,
        "detail":pro_detail,
        "output_Requirement":pro_oReq_text_list
    }
    data_list.append(pro_Data)

    pro_id=row["proId"]
    # 从publicApplication中得到 pdf的接口
    url_pdf='https://summer-ospp.ac.cn/api/publicApplication'
    pdf_data={"proId":pro_id}
    pdf_headers = {
        'Cookie': "tgt=1765368373.029.798.634052|2125b4d7ab6829ae59e8509cd0133f66; UM_distinctid=19b08279f765a8-0f16fe0ea812358-4c657b58-10e280-19b08279f771316; cna=8cbb0cfac97f4ab898233e4291b2eb89; CNZZDATA1281243141=645153189-1765368373-https%253A%252F%252Fgithub.com%252F%7C1765416435",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
        "Content-Length":"15"
    }
    response_pdf=requests.post(url=url_pdf,headers=headers,data=pdf_data)
    pdf_content=response_pdf.content


    pro_name_std=re.sub(r"[<>:/\"\'\\?*`]","_",str(pro_name)).strip()#去掉非法字符和空字符

    dir_2=f"项目申请书/{pro_name_std}"
    if not os.path.exists(dir_2):
        os.mkdir(dir_2)
    pdf_save_path=f"项目申请书/{pro_name_std}/项目申请书.pdf"
    with open(pdf_save_path,'wb') as f:
        f.write(pdf_content)
        print("项目申请书保存成功！！！")


with open("ospp_pro_data.csv","w+",newline='',encoding="utf-8") as f:
    filednames=['name',"diffculty","techTag","detail","output_Requirement"]
    writer=csv.DictWriter(f,filednames)
    writer.writeheader()
    writer.writerows(data_list)