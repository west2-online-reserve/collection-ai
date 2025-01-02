from lxml import etree
import requests
import re
import csv
def trend(url):
    response=requests.get(url)
    html=response.content #减少乱码的可能性
    tree = etree.HTML(html)

    li_list = tree.xpath("//ul[@class = 'list-gl']/li")# /是根节点  //后代  /*/ * ，任意节点是通配符
    text = ""
    list_link = []
    all_title = []
    for li in li_list:
        #result1 = li.xpath("./span/font/text()")[0]
        result = li.xpath(".//text()")
        link = 'https://jwch.fzu.edu.cn/'+li.xpath("./a/@href")[0]# img/@src
        #download = find(link)

        list_link.append(link)


        result = " ".join(result)# result 列表中的元素连接成一个字符串
        # result=re.sub(r"\s+"," ",result)#去除一个或多个空白符
        text1 = re.split(r'\s+', result)#将字符串 text 按照一个或多个空格拆分成一个列表
        text1.append(link)
        text1 = list(filter(None, text1))#删去空元素
        #print(text1)
        #text1 =[result , link ]
        #text = text + result + link +"\n"
        all_title.append(text1)

    # 写入数据到CSV文件
    with open("./title.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 写入行
        writer.writerows(all_title)

    print("所有内容写入完成！")
