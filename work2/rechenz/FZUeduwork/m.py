import requests
from bs4 import BeautifulSoup
import csv
import re
import json


def main():
    url = "https://jwch.fzu.edu.cn/jxtz.htm"
    response = requests.get(url)
    response.encoding = 'utf-8'
    content = response.text
    soup = BeautifulSoup(content, 'lxml')
    check = soup.find('ul', attrs={"class": "list-gl"})
    li_find = None
    if check != None:
        li_find = check.find_all_next('li')
    else:
        print("error")
    with open('AIsolution/work2/FZUeduwork/fzu.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['通知人', '标题', '日期', '链接', '是否存在附件', '附件下载次数'])
        if li_find == None:
            exit()
        for li in li_find:
            annoucer = re.search("【(.*?)】", li.text)
            if annoucer:
                annoucer = annoucer.group(1)
            else:
                annoucer = ""
            temp = li.find('a')
            title = None
            date = None
            href = None
            if temp:
                title = temp.string
                href = temp.get('href')
            temp = li.find('span')
            if temp:
                date = temp.string
                if date == None:
                    tt = temp.find('font')
                    if tt:
                        date = tt.string
            href = str(href)
            href = "https://jwch.fzu.edu.cn/"+href
            curresponse = requests.get(href)
            curresponse.encoding = 'utf-8'
            check = re.search("附件", curresponse.text)
            if check:
                curinfo = BeautifulSoup(curresponse.text, 'lxml')
                download = curinfo.find(
                    'ul', attrs={'style': 'list-style-type:none;'})
                if download:
                    pass
                else:
                    writer.writerow([annoucer, title, date, href, False])
                    continue
                times = download.find_all('span')
                # print(times)
                it = iter(times)
                timesnum = int(0)
                # print(download)
                for i in it:
                    # print(i.string)
                    if i.string == None:
                        continue
                    textid = re.search(r'\(.*?\,', i.string)
                    if textid == None:
                        continue
                    textid = textid.group(0)
                    # print(textid[1:-1])
                    clickurl = "https://jwch.fzu.edu.cn/" + \
                        f'system/resource/code/news/click/clicktimes.jsp?wbnewsid={textid[1:-1]}&owner=1744984858&type=wbnewsfile&randomid=nattach'
                    # print(i.get_text())
                    tempjson = requests.get(clickurl)
                    tempjson = json.loads(tempjson.text)
                    # print(tempjson)
                    timesnum = max(int(tempjson['wbshowtimes']), timesnum)
                writer.writerow([annoucer, title, date, href, True, timesnum])
            else:
                writer.writerow([annoucer, title, date, href, False])
            # writer.writerow([annoucer, title, date, href])
        #     exit()
        # exit()
        for page in range(1, 206, 1):
            url = f"https://jwch.fzu.edu.cn/jxtz/{206-page}.htm"
            # writer.writerow([1, 1, 1, 1, 1])
            print(page)
            response = requests.get(url)
            response.encoding = 'utf-8'
            content = response.text
            soup = BeautifulSoup(content, 'lxml')
            check = soup.find('ul', attrs={"class": "list-gl"})
            li_find = None
            if check != None:
                li_find = check.find_all_next('li')
            else:
                print("error in for")
            if li_find == None:
                exit()
            for li in li_find:
                annoucer = re.search("【(.*?)】", li.text)
                if annoucer:
                    annoucer = annoucer.group(1)
                else:
                    annoucer = ""
                temp = li.find('a')
                title = None
                date = None
                href = None
                if temp:
                    title = temp.string
                    href = temp.get('href')
                temp = li.find('span')
                if temp:
                    date = temp.string
                    if date == None:
                        tt = temp.find('font')
                        if tt:
                            date = tt.string
                href = str(href)
                href = str.replace(href, "../", "https://jwch.fzu.edu.cn/")
                try:
                    curresponse = requests.get(href)
                except Exception:
                    writer.writerow([annoucer, title, date, href, 'NaN'])
                    continue
                curresponse = requests.get(href)
                curresponse.encoding = 'utf-8'
                check = re.search("附件", curresponse.text)
                if check:
                    curinfo = BeautifulSoup(curresponse.text, 'lxml')
                    download = curinfo.find(
                        'ul', attrs={'style': 'list-style-type:none;'})
                    if download:
                        pass
                    else:
                        writer.writerow([annoucer, title, date, href, False])
                        continue
                    times = download.find_all('span')
                    # print(times)
                    it = iter(times)
                    timesnum = int(0)
                    # print(download)
                    for i in it:
                        # print(i.string)
                        if i.string == None:
                            continue
                        textid = re.search(r'\(.*?\,', i.string)
                        if textid == None:
                            continue
                        textid = textid.group(0)
                        # print(textid[1:-1])
                        clickurl = "https://jwch.fzu.edu.cn/" + \
                            f'system/resource/code/news/click/clicktimes.jsp?wbnewsid={textid[1:-1]}&owner=1744984858&type=wbnewsfile&randomid=nattach'
                        # print(i.get_text())
                        tempjson = requests.get(clickurl)
                        tempjson = json.loads(tempjson.text)
                        # print(tempjson)
                        timesnum = max(int(tempjson['wbshowtimes']), timesnum)
                    writer.writerow(
                        [annoucer, title, date, href, True, timesnum])
                else:
                    writer.writerow([annoucer, title, date, href, False])


if __name__ == '__main__':
    main()
