#使用到的第三方库：requests、lxml、selenium、webdriver_manager
import requests
from lxml import html
import time
import re
import jsonpath
url = 'https://jwch.fzu.edu.cn/jxtz.htm'#首页链接
count=209
getcount=0
altime=0
print("正在爬取首页")
with open('nifu.csv','w+',encoding='utf-8') as f:
    while count>=0: #可自定义爬取数量 
        
        starttime=time.time()
        res=requests.get(url)                                      #获取网页内容
        tree = html.fromstring(res.content.decode('utf-8'))
        urlpool=tree.xpath("//ul[@class='list-gl']//a/@href")      #提取链接                        
        usedurlpool=["https://jwch.fzu.edu.cn/"+res.group() for i in urlpool if (res:=re.search(r'info/\d+/\d+(\.)htm|content.+',i))]  #筛选链接  
        Notifierpool=tree.xpath("//ul[@class='list-gl']//li/text()")#提取通知人文本      
        partern=r'质量办|教学运行|电教中心|实践科|综合科|计划科|教研教改|教材中心|铜盘校区管理科'  #定义正则表达式 
        cleaned_Notifierpool=[ res.group() for i in Notifierpool if (res:=re.search(partern, i))]    #提取通知人
        lastpool=dict(zip(usedurlpool, cleaned_Notifierpool)) #将链接和通知人对应起来
        for URL,notifier in lastpool.items():
            response=requests.get(URL)
            if response!=None:           
                tree = html.fromstring(response.content.decode('utf-8'))
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
                getcount+=1
                f.write(f'--------------第{getcount}条--------------\n')#储存数据
                f.write('来源:'+str(notifier)+'\n')            
                f.write(titles[0] + '\n')
                f.write(date[0] + '\n')
                f.write('详情链接: '+URL+'\n')
                f.write('通知人:'+str(Notifier[0]) + '\n')
                for i in range(len(attachment)):
                    f.write('附件:'+attachment[i]+'\n')
                    f.write('下载链接:'+"https://jwch.fzu.edu.cn"+attachmenturl[i]+'\n')
                    f.write('下载次数:'+str(clicktimes[i])+'\n')
                    # attachmentobj=requests.get("https://jwch.fzu.edu.cn"+attachmenturl[i])
                    # print("正在下载附件:"+attachment[i])
                    # with open(attachment[i], 'wb') as f2:  #下载附件
                    #     f2.write(attachmentobj.content)
                    
        endtime=time.time() 
        altime+=(endtime-starttime)
        print("耗时",f"{(endtime-starttime):.6f}","秒")
        url='https://jwch.fzu.edu.cn/jxtz'+f"/{count}.htm"  #获取下一页链接，翻页
        print("正在爬取第"+str(count)+"页")
        count-=1   

print("总耗时",f"{altime:.6f}","秒")