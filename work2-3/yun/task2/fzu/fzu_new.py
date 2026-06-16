#使用到的第三方库：requests、lxml、selenium、webdriver_manager
import requests
from lxml import html
import time
import re
import jsonpath
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import threading

#利用多线程优化爬取和分析数据的效率，使用队列实现线程间通信，确保数据的安全性和完整性，测试时长总耗时 434.415154 秒，而单线程爬取总耗时 1720.312682 秒 ,效率提升约4倍(容易被ban)
#先一页一页爬取URL和通知人列表，将其扔进线程池获取具体通知内容，分析完成后将结果放入用于保存url、通知人、具体内容的队列中
#之后push_data从队列中获取URL、通知人和具体内容，分析完成后将结果放入待保存队列中，最后save_data从待保存队列中获取分析结果并保存到文件中

def getpage(url):
    res=requests.get(url)                                  #获取网页内容
    tree = html.fromstring(res.content.decode('utf-8'))
    urlpool=tree.xpath("//ul[@class='list-gl']//a/@href")      #提取链接    
    usedurlpool=["https://jwch.fzu.edu.cn/"+res.group() for i in urlpool if (res:=re.search(r'info/\d+/\d+(\.)htm|content.+',i))]  #筛选链接 
    Notifierpool=tree.xpath("//ul[@class='list-gl']//li/text()")#提取通知人文本
    partern=r'质量办|教学运行|电教中心|实践科|综合科|计划科|教研教改|教材中心|铜盘校区管理科'  #定义正则表达式 
    cleaned_Notifierpool=[ res.group() for i in Notifierpool if (res:=re.search(partern, i))]    #提取通知人
    lastpool=zip(usedurlpool, cleaned_Notifierpool) #将链接和通知人对应起来 
    return lastpool

def getdata(url,notifier):
    time.sleep(0.7) #等待0.7秒，模拟网络延迟
    request=requests.get(url) #获取网页内容
    if request!=None:
        return url,notifier,request.content.decode('utf-8')            #返回URL、通知人和网页内容
    
def analysis_data(URL,notifier,request):
            tree = html.fromstring(request)
            titles=tree.xpath("//h4/text()")                           #提取标题
            attachment=tree.xpath("//li/a[@href]/text()")              #提取附件标题
            attachmenturl=tree.xpath("//li/a/@href")                   #提取附件链接
            wbnewsid=tree.xpath("//li/span[@id]/@id")                  #提取id
            cleaned_wbnewsid=[res.group() for i in wbnewsid if (res:=re.search(r'\d+', i))] #提取数字部分作为wbnewsid
            clicktimes=[]
            for i in cleaned_wbnewsid:
                content=requests.get(f'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid={i}&owner=1744984858&type=wbnewsfile&randomid=nattach')   #获取点击次数的网页内容
                clicktimes.append(jsonpath.jsonpath(content.json(), '$.wbshowtimes')[0])  #提取点击次数
            date=tree.xpath("//span[@class='xl_sj_icon']/text()")      #提取发布时间
            signature=[tree.xpath(f"string(//div[@class='v_news_content']//p[last()-{i}])") for i in range(15) ]   #提取落款
            Notifier=[i.strip().replace('\n', '').replace('\t', '').replace('\r', '') for i in signature if re.search(r'教学质量监控与评估中心|教务处|教学运行科|注册中心', i)]    #提取通知人
            if Notifier==[]:#如果在//div[@class='v_news_content']//p[last()-{i}])中没有找到通知人，则在//div[@class='v_news_content']/p[last()-{i}]中寻找
                signature=[tree.xpath(f"string(//div[@class='v_news_content']/p[last()-{i}])") for i in range(15) ]   #提取落款
                Notifier=[i.strip().replace('\n', '').replace('\t', '').replace('\r', '') for i in signature if re.search(r'教学质量监控与评估中心|教务处|教学运行科|注册中心', i)]    #提取通知人
                if Notifier==[]:
                    Notifier=['未找到通知人']
            return URL,notifier,titles[0],date[0],Notifier[0],attachment,attachmenturl,clicktimes

def fetch(maxsize=10):
    URL,notifier,request=[],[],[]
    
    while not q.empty() and len(URL) < maxsize:
        _URL,_notifier,_request=q.get(timeout=5) #从队列中获取URL、通知人和网页内容
        URL.append(_URL)                #将URL添加到列表中
        notifier.append(_notifier)      #将通知人添加到列表中
        request.append(_request)        #将网页内容添加到列表中
    return URL,notifier,request

def gethtml(startpage, url):
    count=startpage
    with ThreadPoolExecutor(max_workers=10) as executor: #创建线程池，最大线程数为10
        while count>=0: #可自定义爬取数量
            lastpool=getpage(url) #获取当前页的URL和通知人列表
            futures=[executor.submit(getdata,url,notifier) for url,notifier in lastpool] #将爬取函数和参数传入线程池，异步执行
            for future in futures:
                    URL,notifier,request=future.result() #获取爬取结果
                    q.put((URL,notifier,request)) #将爬取结果放入队列中                                                                                                         
            url='https://jwch.fzu.edu.cn/jxtz'+f"/{count}.htm"  #获取下一页链接，翻页
            print("正在爬取第"+str(count)+"页")
            count-=1 

def analysisandpush_data():
     with ThreadPoolExecutor(max_workers=10) as executor: #创建线程池，最大线程数为10
            while not q.empty():
                print(f"还有{q.qsize()}个网页待分析")                                   #打印队列中剩余的URL数量
                URL,notifier,request=fetch()                                           #从队列中批量获取URL、通知人和网页内容
                futures=[executor.submit(analysis_data,_URL,_notifier,_request)  for _URL,_notifier,_request in zip(URL,notifier,request)] #将分析函数和参数传入线程池，异步执行
                for future in futures:
                    URL,notifier,titles,date,Notifier,attachment,attachmenturl,clicktimes=future.result() #获取分析结果
                    q_data.put((URL,notifier,titles,date,Notifier,attachment,attachmenturl,clicktimes))   #将分析结果放入另一个队列中
                if q.empty():
                    print("等待爬取")
                    time.sleep(3) #队列为空,等待3秒，
            print("数据分析完毕")

def save_data():
        getcount=1
        with open('fzu.csv','w+',encoding='utf-8') as f:
            while not q_data.empty():
                print(f"还有{q_data.qsize()}个结果待保存") #打印队列中剩余的分析结果数量
                URL,notifier,titles,date,Notifier,attachment,attachmenturl,clicktimes=q_data.get(timeout=5) #从队列中获取分析结果
                f.write(f'--------------第{getcount}条--------------\n')#储存数据
                f.write('来源:'+str(notifier)+'\n')            
                f.write(titles + '\n')
                f.write(date + '\n')
                f.write('详情链接: '+URL+'\n')
                f.write('通知人:'+str(Notifier) + '\n')
                for i in range(len(attachment)):
                    f.write('附件:'+attachment[i]+'\n')
                    f.write('下载链接:'+"https://jwch.fzu.edu.cn"+attachmenturl[i]+'\n')
                    f.write('下载次数:'+str(clicktimes[i])+'\n')
                    # attachmentobj=requests.get("https://jwch.fzu.edu.cn"+attachmenturl[i])
                    # print("正在下载附件:"+attachment[i])
                    # with open(attachment[i], 'wb') as f2:  #下载附件
                    #     f2.write(attachmentobj.content)
                getcount+=1
                time.sleep(0.1) #等待0.1秒，模拟文件写入时间
                if q_data.empty():
                    print("等待分析")
                    time.sleep(5) #队列为空,等待10秒
            print("数据保存完毕")         

q=Queue()
q_data=Queue()
url = 'https://jwch.fzu.edu.cn/jxtz.htm'#首页链接

def main():
    startpage=212
    starttime=time.time() 
    print("正在爬取首页")
    
    task1_gethtml=threading.Thread(target=gethtml,args=(startpage, url)) #创建线程
    task2_analysis_data=threading.Thread(target=analysisandpush_data) #创建线程
    task3_save_data=threading.Thread(target=save_data) #创建线程


    task1_gethtml.start()
    time.sleep(5) #等待5秒，确保爬取线程已经开始爬取数据
    print("正在分析数据")
    task2_analysis_data.start()
    time.sleep(5) #等待5秒，确保分析线程已经开始分析数据
    print("正在保存数据")
    task3_save_data.start()


    task1_gethtml.join()
    task2_analysis_data.join()
    task3_save_data.join()


    endtime=time.time()
    print("总耗时",f"{(endtime-starttime):.6f}","秒")

if __name__ == '__main__':
    main()
