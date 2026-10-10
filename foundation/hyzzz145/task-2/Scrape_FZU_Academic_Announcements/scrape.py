import requests
import csv
import re
from bs4 import BeautifulSoup 
import os
#通过分析福大教务处通知的 url，可以掌握规律，第一个数字代表模块，第二个数字代表内容 id
#但是这个 id 好像毫无规律，如（14808，14802，14785，14755）
#我只好一个一个遍历，用 if 判断 id 是否存在
#我发现我是猪b，为什么我不直接在教学通知里直接获得相应链接
#头文件
head = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36 Edg/154.0.0.0"
}
#初始化
idx = 0
end = 150
run = 1
##进入程序文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))





#定义爬取函数
def scatch(resp_notice,url_id):
    global idx
    global run,writer,f
    #检验是否正常运行
    if resp_notice.status_code == 200:
        #中文异常解决
        resp_notice.encoding = resp_notice.apparent_encoding
        notice_text = resp_notice.text
        #获取正文链接
        link_re = re.compile(r'<a href=".*?(?P<link>info.*?)".*?</a>',re.S)
        link_context = link_re.finditer(notice_text)
        #爬取data1的正则
        context_re = re.compile(r'<a href="../../jxtz/.*?\.htm">(?P<part>.*?)</a>.*?'
                            r'<h4 align="center">(?P<title>.*?)</h4>.*?'
                            r'<span class="xl_sj_icon">.*?(?P<date>[\d\-]+)</span>.*?'
                            ,re.S)
        #附件正则
        context_re_2 = re.compile(r'<a href="(?P<ex_file_link>/system/_content/download.jsp\?urltype=news.DownloadAttachUrl.owner=1744984858&wbfileid=(?P<ex_file_dlnum_id>.*?))" target="_blank">(?P<ex_file_name>.*?)</a>.*?',re.S)
        #遍历通知页正文链接（20个）
        for i in link_context:
            #记录第几个文件
            idx += 1
            print(f"第{idx}个文件")
            #内容筛选
            url_context = "https://jwch.fzu.edu.cn/" + str(i.group("link"))
            #检查超时
            try:
                resp_context = requests.get(url_context,headers=head,timeout=3)
            except (requests.Timeout,requests.ConnectionError) as e:
                with open("log.txt",'w') as log:
                    log.write(url_id)
                print(f"网路中断;{e}")
                run = 0

            #是否正常
            if resp_context.status_code == 200:
                resp_context.encoding = resp_context.apparent_encoding
                context_text = resp_context.text
                data = context_re.finditer(context_text)
                data_2 = context_re_2.finditer(context_text)
                #data 1
                for o in data:
                    part = o.group("part")
                    title = o.group("title")
                    date = o.group("date")
                    writer.writerow([f"通知人：{part}"])
                    writer.writerow([f"标题：{title}"])
                    writer.writerow([f"发布时间：{date}"])
                    writer.writerow([f"详细链接：{url_context}"])
                    #正文（略去）
                    soup = BeautifulSoup(context_text,'lxml')
                    content = soup.find('div',class_ = 'xl_main').get_text(strip=True)
                    writer.writerow([f"正文：{content}"])


                    print("data1 已写入")
                    writer.writerow(["附件信息：  "])
                #data 2 是否存在
                if next(data_2,None) == None:
                    print("false")
                    writer.writerow(["\t无"])
                    writer.writerow([" "])
                else:
                    #遍历data 2
                    for p in data_2:
                        ex_file_name = p.group("ex_file_name")
                        ex_file_link = "https://jwch.fzu.edu.cn/" + p.group("ex_file_link")
                        #爬取下载次数
                        url_dlnum = f"https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={p.group("ex_file_dlnum_id")}&owner=1744984858&type=wbnewsfile&randomid=nattach"
                        resp_dlnum = requests.get(url_dlnum)
                        dlnum = resp_dlnum.json()["wbshowtimes"]
                        writer.writerow([f"\t附件名：{ex_file_name}"])
                        writer.writerow([f"\t附件链接：{ex_file_link}"])
                        writer.writerow([f"\t已下载{dlnum}次"])
                        writer.writerow([" "])
                        print("data2已写入")
                        print("")
            else:
                print("异常")
    #防止中途网络终端
    else:
        with open("log.txt",'w') as log:
            log.write(url_id)
        print("连接异常")
        run = 0
        

#定义通知页请求函数
def notice_request(url_id):
    global run
    with open("log.txt",'w') as log:
        log.write(url_id)
    print(f"url_id={url_id}")
    url_notice = "https://jwch.fzu.edu.cn/jxtz" + url_id + ".htm"
    #发送请求
    try:
        resp_notice = requests.get(url_notice)
    except (requests.Timeout,requests.ConnectionError) as e:
        with open("log.txt",'w') as log:
            log.write(url_id)
        print(f"网路中断;{e}")
        run = 0
    return resp_notice
   
def run_code():
    global run,writer,url_id,f
    while run:
        #重新开始 
        if not is_continue:
            #打开csv，以及初始化覆盖
            f = open("content.csv",'w',newline="",encoding='utf-8')
            writer = csv.writer(f)
            #遍历首页通知网址
            for url_id in ([""] + [f"/{n}" for n in range(215, end, -1)]):  
                scatch(notice_request(url_id),url_id)
        else:
            #打开csv，以及初始化追加模式
            f = open("content.csv",'a',newline="",encoding='utf-8')
            writer = csv.writer(f)
            #读取报告
            with open("log.txt",'r') as log_id:
                url_id_log = log_id.read()
            if url_id_log == "":
                for url_id in ([""] + [f"/{n}" for n in range(215, end, -1)]): 
                    scatch(notice_request(url_id),url_id)
            else:
                for url_id in [f"/{n}" for n in range(int(re.search(r'\d+',url_id_log).group())-1, end, -1)]:
                    scatch(notice_request(url_id),url_id)
        run = 0
        f.close()




        

        #附件

    


if __name__ == '__main__':
    #询问是否从上次为保存的继续
    is_continue = False
    is_continue = bool(int(input("是否从上次为保存的继续(1/0)")))
    run_code()
print("over")





