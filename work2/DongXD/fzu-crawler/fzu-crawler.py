import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

nums=500
news_paths=[]

host='https://jwch.fzu.edu.cn/'
path='jxtz.htm'

print(f"获取资源 {host+path}")
response=requests.get(host+path)
soup=BeautifulSoup(response.content,'lxml')

while True :
    print(f"正在解析 {host+path}")

    next_page_path=soup.find('span',class_='p_next p_fun').find('a').get('href')
    news=soup.find('ul',class_='list-gl').find_all('li')

    for new in news:
        news_path=new.find('a').get('href')

        if news_path[0]=='.':
            news_path=news_path[2:]

        news_paths.append(news_path)

        if len(news_paths)==nums:
            break
    else:
        print(f"进度 {len(news_paths)}/{nums}")
        path=next_page_path

        if path[0]!='j':
            path="jxtz/"+path

        print(f"获取资源 {host+path}")
        response=requests.get(host+path)
        soup=BeautifulSoup(response.content,'lxml')
        continue
    break

datas=[]

def fetch_news_data(news_path) :
    response=requests.get(host+news_path)
    soup=BeautifulSoup(response.content,'lxml')

    author=soup.find('p', class_='w-main-dh-text').find_all('a')[-1].text
    title=soup.find('h4').text
    date=soup.find('span',class_="xl_sj_icon").text[5:]

    appendixes_htms=[x  for x in soup.find_all('li') if x.text[:2]=="附件"]
    appendixes=[]

    for appendixes_htm in appendixes_htms:
        name=appendixes_htm.text[2:-5]
        ps=appendixes_htm.find("script").text[14:-1].split(',')

        api="https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp"
        params = {
            "wbnewsid": ps[0],
            "owner": ps[1],
            "type": ps[2][1:-1],
            "randomid": ps[3][1:-1]
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
        }
        response=requests.get(api,params=params,headers=headers)

        dltime=response.json()['wbshowtimes']
        dlurl=appendixes_htm.find('a').get('href')
        appendixes.append({
            "附件名":name,
            "下载词数":dltime,
            "链接码":dlurl
        })

    print(f"完成解析 {host+news_path}")
    return {
        "通知人":author,
        "标题":title,
        "日期":date,
        "详细链接":host+news_path,
        "附件":appendixes
    }

with ThreadPoolExecutor(max_workers=30) as executor:
    future_to_path ={
        executor.submit(fetch_news_data,path): path for path in news_paths
    }

    for future in as_completed(future_to_path):
        news_path=future_to_path[future]
        data=future.result()
        datas.append(data)
        print(f"进度{len(datas)}/{nums}")

df=pd.DataFrame(datas)
print(df)
df.to_csv("data.csv")

    
        






