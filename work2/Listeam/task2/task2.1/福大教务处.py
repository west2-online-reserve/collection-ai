import requests
import csv
from lxml import etree
from pprint import pprint
import re
import json

def get_urls():
    headers = {
"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
}
    Urls = []
    informs_list =[]
    ajax_prefix = 'https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp?wbnewsid=' #地址太长，把前后缀提出来，简洁代码
    ajax_suffix = '&owner=1744984858&type=wbnewsfile&randomid=nattach'
    max_attach = 0
    base_domain = "https://jwch.fzu.edu.cn/"  #泥福超绝省略前面地址，搞个模板来拼接
    for i in range(181,206,1): #获取中间四十页的通知
        response = requests.get(f"https://jwch.fzu.edu.cn/jxtz/{i}.htm",headers = headers)
        text = response.text
        text_html = etree.HTML(text)
        links = text_html.xpath('//div[@class="box-gl clearfix"]//li/a/@href')
        for link in links:   #提取每页每个通知的链接
            integrated_link = requests.compat.urljoin(base_domain,link) #用compat的urljoin方法拼接基础域名和省略后的链接
            Urls.append(integrated_link)
    #print(Urls)

    for url in Urls:
        response = requests.get(url,headers = headers)
        response.encoding = response.apparent_encoding #根据网址自动识别编码防止中文乱码
        text = response.text
        text_html = etree.HTML(text)
        title = text_html.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/div[1]/h4/text()')[0].strip("'")

        notifier_list = text_html.xpath('/html/body/div/div[2]/div[1]/p/a[3]/text()')
        notifier = notifier_list[0].strip("'") if notifier_list else '未知' #先提取列表检测非空，防止报错

        origin_date = text_html.xpath('/html/body/div/div[2]/div[2]/form/div/div[1]/div/div[2]/div[1]/span[1]/text()')[0].strip("'")
        final_date = origin_date.split('：')[1] #用split一刀两断形成列表并提取第二项，去除中文

        accessory_name = re.findall(r'<a href="/system/_content/download.jsp\?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=\d+" target="_blank"(.+?)</a>',text)  #此处加？是因为不贪多，只到第一个a就停止，如果不加就可能到后面的a才结束

        accessory_link_raw = re.findall(r'href="/system/_content/download.jsp\?urltype=news.DownloadAttachUrl&owner=1744984858&wbfileid=\d+',text)
        accessory_link = [base_domain + link for link in accessory_link_raw]#直接用列表推导式修改
        inform = {

            '通知人': notifier,
            '标题': title,
            '日期': final_date,
            '详情链接': url,
            
        }

        if accessory_name:  #如果此列表非空才运行！！！服了后面才发现一直爬不出来是因为name没出来，不是阿贾克斯错了
            if len(accessory_name) > max_attach:
                max_attach = len(accessory_name)  #不好根据每个附件个数设置表头，直接取附件数最大值
            for i in range(0,len(accessory_name)):  
                ajax_id = re.findall(r'wbfileid=(\d+)',text)[i]  #用正则找到阿贾克斯请求url的唯一可变参量id

                download_url = ajax_prefix + ajax_id + ajax_suffix  #说白了只有下载量需要阿贾克斯，链接和名称可以直接用正则提取出来
                download_dic = requests.get(download_url,headers = headers) #阿贾克斯要重新requests
                download_dic_json = json.loads(download_dic.text)  #json的loads方法解码
                download_counts = str(download_dic_json["wbshowtimes"]) 

                inform.update({
                    f'附件{i+1}':accessory_name[i],
                    f'下载量{i+1}':download_counts,  #下载量需要的id一次只获取一个，因为拼接url不能用列表形式，在前面就得取[]，所以这里不用[]
                    f'附件链接{i+1}':accessory_link[i],
                })
            
        informs_list.append(inform)
        #pprint(informs_list,indent=4)  #特殊打印，间隔四空格
        
        
    with open(r'C:\Users\Lst12\Desktop\福大教务处通知.csv', 'w', encoding='utf-8-sig', newline=' ') as f:
        header = ['通知人', '标题', '日期', '详情链接',]
        for i in range(0,max_attach):
            header.extend([f'附件{i+1}',f'下载量{i+1}',f'附件链接{i+1}'])
        writer = csv.DictWriter(f,header)
        writer.writeheader()
        writer.writerows(informs_list)
    
if __name__ == "__main__":
    get_urls()
    