import requests,os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + r'\file')

def grab_pdf():
    id = int(input('你要抓取pdf的项目id后两位：'))
    refer_url = 'https://summer.ospp.ac.cn/previewPdf/'+f'{id}'
    payload = {
        'proId':f'{id}'
    }
    head = {
        'referer':refer_url,
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0'
    }
    resp = requests.post('https://summer.ospp.ac.cn/api/publicApplication',headers=head,json=payload)
    if resp.status_code == 200:
        print('请求成功')
        run = True
    else:
        print("请求失败")
        run = False
    with open(f'{id}.pdf','wb') as f:
        f.write(resp.content)
    if run:
        print(f'id:{id}成功下载')
    else:
        print(f'id:{id}无法下载')
if __name__ == '__main__':
    grab_pdf()