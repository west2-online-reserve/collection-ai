import requests
import re

def get_pro_detail(response_detail:requests.Response) -> tuple:
    # 获得单个项目的项目简介和产出要求
    text=response_detail.content.decode('utf-8')
    pro_info=re.search(r'"programDesc":"(.*?)","programDescEN"',text).group(1)

    output_info_general=re.search(r'"outputRequirement":(.*?),"outputRequirementEN"',text)
    output_info_list=re.findall(r'"title":"(.*?)"',output_info_general.group(1))
    output_info=''
    for i in output_info_list:
        output_info+=i

    return (pro_info,output_info)    

def get_pro_by_page(general_list:list,response:requests.Response):
    # 将该页面的项目列表信息加入总列表
    text=response.content.decode('utf-8')
    pro_list=re.findall(r'{"programCode":"(.*?)".*?"difficulty":"(.*?)".*?"programName":"(.*?)","orgName":"(.*?)",.*?}',
                       text 
    )

    for i in pro_list:

        json={"programId":i[0],"type":"org"}
        headers={
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/146.0'
        }
        response_detail=requests.post(url='https://summer-ospp.ac.cn/api/getProDetail',headers=headers,json=json)
        detail_tuple=get_pro_detail(response_detail)

        pro_dic={
            'programCode':i[0],
            '项目名称':i[2],
            '项目难度':i[1],
            '社区名称':i[3],
            '项目简述':detail_tuple[0],
            '项目产出要求':detail_tuple[1]
        }
        general_list.append(pro_dic)

        print('finished one item')

def write_csv(general_list:list):
    # 定义CSV表头（与字典的key对应）
    headers = ['项目名称','项目难度','社区名称','项目简述','项目产出要求']
    
    from pathlib import Path
    # 一行获取脚本所在文件夹（Path 对象，可直接拼接）
    current_script_dir = Path(__file__).parent

    import csv
    # 直接拼接文件路径（无需手动加分隔符）
    file_path = current_script_dir / "pro_list.csv"  # / 是路径拼接符
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        # DictWriter按表头写入，自动对齐列
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()  # 写入表头

        general_list_tmp=[]
        for i in general_list:
            dict_tmp={
                '项目名称':i['项目名称'],
                '项目难度':i['项目难度'],
                '社区名称':i['社区名称'],
                '项目简述':i['项目简述'],
                '项目产出要求':i['项目产出要求']
            }
            general_list_tmp.append(dict_tmp)

        writer.writerows(general_list_tmp)  # 写入所有数据行
   
if __name__=='__main__':
    url='https://summer-ospp.ac.cn/api/getProList'
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/146.0'
    }

    pro_dict_list=[]
    for page_num in range(1,13):
        json={
            "supportLanguage":[],
            "techTag":[],
            "programmingLanguageTag":[],
            "programName":"",
            "difficulty":[],
            "pageNum":f'{page_num}',
            "pageSize":'50',
            "lang":"zh",
            "orgName":[]
        }

        # 抓包项目列表
        response=requests.post(url=url,headers=headers,json=json)

        get_pro_by_page(pro_dict_list,response)
    
        print(page_num,'页')

    # 写入csv
    write_csv(pro_dict_list)
    