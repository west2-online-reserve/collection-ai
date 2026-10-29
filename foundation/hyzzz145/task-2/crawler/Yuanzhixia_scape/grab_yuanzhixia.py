import requests,traceback,re,csv,os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#初始化
with open('log.txt','r') as f:
    page_num = int(f.read())
#头文件
header = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0',
}
#网址
url_homepage = 'https://summer.ospp.ac.cn/org/projectlist?lang=zh&pageNum=1&pageSize=50'
url_xhr_getprogarmid = 'https://summer.ospp.ac.cn/api/getProList'
#把列表转为字符串
def list_into_str(list)->str:
    stra = ""
    for i in list:
        stra += str(i)
    return stra

#获得项目id的函数
def get_progarmid(xhr:str,refer:str)->list:
    json_rows_list = []
    #payload
    payload = {"pageSize": 50,
            "pageNum": 2,}
    try:
        head = header
        head['referer'] = refer
        resp = requests.post(xhr,headers=head,json=payload)
        json_list = resp.json()
        json_rows = json_list['rows']
        for i in range(46):
            print(i)
            json_rows_list.append(json_rows[i]['programCode'])
        print('已获取id')
        # for i in range(50):
        #     programCode = json_rows[i]['programCode']
        #     url_xhr_content = r'https://summer.ospp.ac.cn/api/getProDetail'

    except Exception as e:
        print("异常类型：", type(e).__name__)
        print("异常内容：", repr(e))
        traceback.print_exc()
    return json_rows_list
#获得目标json字典
def get_content(programCode:str)->dict:
    payload = {
        'programId':f'{programCode}'
    }
    print(programCode)
    try:
        url_xhr_content = r'https://summer.ospp.ac.cn/api/getProDetail'
        url_referer_content = f'https://summer.ospp.ac.cn/org/prodetail/{programCode}?lang=zh&list=pro'
        head = header
        head['referer'] = url_referer_content
        print("准备发起resp")
        resp = requests.post(url_xhr_content,headers=head,json=payload)
        # print(resp.status_code)
        # print("状态码：", resp.status_code)
        # print("请求方法：", resp.request.method)
        # print("请求网址：", resp.request.url)
        # print("请求体：", resp.request.body)
        # print("响应类型：", resp.headers.get("Content-Type"))
        # print("响应正文：", repr(resp.text[:1500]))
        json = resp.json()
        #正则匹配领域
        re_field = re.compile(r'datas\",\"(?P<field>.*?)\"')
        field_iter = re_field.finditer(json['techTag'])
        field_list = []
        for i in field_iter:
            field_list.append(i.group('field') + ' ')
        #获得产出要求列表
        list_outrequire = []
        list_outrequire_dict = json['outputRequirement']
        list_outrequire_dict.pop(0)
        for i in list_outrequire_dict:
            list_outrequire.append(i['title'])
        json_dict = {
            'title':json['programName'],
            'difficulty':json['difficulty'],
            'tech_field':list_into_str(field_list),
            'detail_introduce':re.sub(r'<[^>]*>','',json['programDesc']).strip(),
            'detail_outrequire':list_into_str(list_outrequire)
                     }
    except Exception as e:
        print('错误类型：',type(e).__name__)
        print('错误内容：',repr(e))
        traceback.print_exc()
    return json_dict
    #定义写入csv函数
def write_csv(dic:dict):
    # while is_again != 2:
    #     is_again = input("是否重写（0/1）")
    #     if is_again != 0 or is_again != 1:
    #         print('请输入合法值')
    # else:
    #     pass
    # if input("是否重写（0/1）") == 1:
    #     with open('yuanzhixia.csv','w')as f:
    #         f.write('')
    #     #防止程序再次重写
    #     is_again = 2
    f = open('yuanzhixia.csv','a',encoding='utf-8',newline="")
    write = csv.writer(f)
    write.writerow([f'--------第{page_num}份---------'])
    for key,value in dic.items():
        write.writerow([f"{key}：{value}"])
    write.writerow('')
    f.close()
    
if __name__ == '__main__':
    code = get_progarmid(url_xhr_getprogarmid,url_homepage)
    # print(get_content(code[0]))
    #是否重写
    is_again = int(input("是否重写（0/1）"))
    while not (is_again == 0 or is_again == 1):
        print('请输入合法值')
        is_again = int(input("是否重写（0/1）"))
    if is_again == 1:
        with open('yuanzhixia.csv','w')as f:
            f.write('')
        page_num = 0
        with open('log.txt','w') as f:
            f.write(str(page_num))
            
    for id in code:
        page_num += 1
        write_csv(get_content(id))
    with open('log.txt','w') as f:
        f.write(str(page_num))
    print('over')

