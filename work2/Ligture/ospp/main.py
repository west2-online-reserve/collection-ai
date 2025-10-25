import requests
import json
import csv


session = requests.Session()
session.headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
}
session.trust_env = False  # 绕过小猫咪

pro_list_api = 'https://summer-ospp.ac.cn/api/getProList'
pro_detail_api = 'https://summer-ospp.ac.cn/api/getProDetail'

writer = csv.writer(open('program.csv', 'w', newline='',encoding='utf-8'))

def download_pdf(pro_id:str,filename):
    pdf_content = session.post('https://summer-ospp.ac.cn/api/publicApplication',data={'proId':pro_id})
    with open(filename, 'wb') as f:
        f.write(pdf_content.content)



def fetch_pro_detail(pro_id):
    detail_data = {
        'programId': pro_id,
        'type': 'org'
    }
    pro_detail = json.loads(session.post(pro_detail_api,data=detail_data).content)
    language = [a.strip('\\') for a in pro_detail['programmingLanguageTag']]
    tech = json.loads(pro_detail['techTag'])
    tech = ','.join([a[1] for a in tech])
    output_requirement_text = ''
    for i in pro_detail['outputRequirement']:
        if i is not None:
            title = i['title'].strip('')
            output_requirement_text += title+'\n'
            if len(i['children']) > 0:
                for child in i['children']:
                    output_requirement_text += ' - ' + child.strip('')+'\n'
    detail = {
        'language': language,
        'tech': tech,
        'desc': pro_detail['programDesc'],
        'difficulty': pro_detail['difficulty'],
        'output_requirement': output_requirement_text,
        'name': pro_detail['programName'],
        'orgProgramId': pro_detail['orgProgramId'],
    }
    return detail


def main():
    data = {
        'difficulty': [],  # "不限","基础/Basic", "进阶/Advanced"
        'lang': "zh",
        'orgName': "",
        'pageNum': 1,
        'pageSize': 50,
        'programName': "",
        'programmingLanguageTag': [],
        'supportLanguage': [],
        'techTag': [],
    }
    pro_list = json.loads(session.post(pro_list_api, data=data).content)
    writer.writerow(['项目名','项目难度','技术领域','项目简述','项目产出要求'])

    for i in pro_list['rows']:
        pro_code = i['programCode']
        p_detail = fetch_pro_detail(pro_code)
        writer.writerow([p_detail['name'],p_detail['difficulty'],p_detail['tech'],p_detail['desc'],p_detail['output_requirement']])
        download_pdf(p_detail['orgProgramId'],f'.\\pdf\\{p_detail['name']}.pdf')


if __name__ == '__main__':
    main()