from bs4 import BeautifulSoup
import requests
import re
import os
import csv
import json

# 基础数据
COUNT=0
base_url="https://jwch.fzu.edu.cn/"
url='https://jwch.fzu.edu.cn/jxtz.htm'
#创建一下第三小问存储HTML文件以及附件的文件夹
folder_name="Html_and_Attachment_of_Each_Notice"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#初始化要写入CSV文件的字典列表
all_data=[]

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
    }

#翻页功能
n=int(input("请输入你需要爬取的数据数量 (限定为20的整数倍，且建议不超过4000条):"))
if n%20:
    n=n//20+1
else:
    n=n//20
new_url=None
for i in range(n):

    if not(new_url is None):
        url=new_url
    print(url)

    response = requests.get(url,headers=headers)
    response.encoding="utf_8"
    content=response.text
    soup=BeautifulSoup(content,'html.parser')

    #获取下一个 url，即 new_url
    nu_span = soup.find("span",attrs={"class":"p_next p_fun"})
    new_url_tail=nu_span.find("a")["href"]
    if new_url_tail[:5]!="jxtz/":
        new_url_tail="jxtz/"+new_url_tail
    new_url=base_url+new_url_tail
    print(new_url)
    #<span class="p_next p_fun">
    # <a href="205.htm">下页</a>
    # </span>
    ul_list = soup.find("ul",attrs={"class":"list-gl"})
    li_list = ul_list.find_all("li")

    for li in li_list: # 对每个通知进行初步处理
        #初始化要写入总数据列表的字典
        ele_data={}
        #Key值有这些：['通知人','标题','日期','详情链接','附件名','附件下载次数','附件链接码']

        # 2.提取通知信息中的「通知人」（如：质量办、计划科）、标题、日期、详情链接。
        a_tag=li.find("a")

        title=a_tag.text.strip() # 标题
        # 要清除文件名中的非法字符，否则后续会报错
        illegal_chars = r'[\\/*?:"<>|]'  # Windows 的非法字符集
        cleaned_title = re.sub(illegal_chars, '_', title) # 将所有非法字符替换为下划线
        ele_data['标题']=cleaned_title

        link=base_url+a_tag.attrs["href"] # 链接
        ele_data['详情链接']=link
        time=li.find("span",attrs={"class":"doclist_time"}).text.strip() # 日期
        ele_data['日期']=time
        temp_noticer=re.findall(r'【.+?】',li.text.strip())
        noticer=re.sub(r'[【】]', '', temp_noticer[0]) # 通知人
        ele_data['通知人']=noticer
        #print(title,link,time,noticer)

        # 下面对每个list进行处理
        # 3.爬取通知详情的 HTML，可能存在「附件」，提取附件名、附件下载次数、附件链接码，有能力请尽可能将附件爬取下来。
        # 先在刚开始创建的文件夹下面再建一个次级文件夹，来存储爬下来的HTML文件和附件
        full_path = os.path.join(folder_name, time+' '+cleaned_title) #HTML文件及附件文件夹
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        #存储对应的HTML文件
        response=requests.get(link,headers=headers)
        file_path=os.path.join(full_path,cleaned_title+'.html')
        with open(file_path,'w+',encoding='utf_8') as res:
            # 这里要手动解码，直接在文件里写入 response.text 的话会产生乱码
            html_content = response.content.decode('utf_8')
            res.write(html_content)
        #现在尝试处理每个HTML文件，尝试获取附件信息
        #先初始化字典数据
        ele_data['附件名']="None"
        ele_data['附件下载次数']="None"
        ele_data['附件链接码']="None"
        have_attach=1
        try:
            COUNT += 1
            print(f"\n正在尝试第{COUNT}个通知")
            subsoup = BeautifulSoup(response.content, 'html.parser')
            div=subsoup.find("div",attrs={"class":"xl_main"})
            if not div:
                have_attach=0
            ul=div.find("ul")
            if not ul:
                have_attach=0
            li=ul.find("li")
            if not li:
                have_attach=0
            #链接码
            attach_link_tail=li.find("a")["href"] #附件链接的后半部分
            attach_link = base_url + attach_link_tail  # 附件链接
            ele_data['附件链接码'] = attach_link
            #附件名
            attach_name=li.find("a").text.strip() #附件名称
            ele_data['附件名'] = attach_name

            #附件下载次数
            #这里的附件下载次数是从getClickTimes(a,b,c,d)函数里面动态获取的，要特别处理

            attach_download_times_original_string=li.find("span").string #附件下载次数的初始字符串
            pattern_or=r"getClickTimes\((\d+),(\d+),\"([^\"]*)\",\"([^\"]*)\"\)"
            match = re.search(pattern_or,attach_download_times_original_string)
            param_or={
                "wbnewsid":int(match.group(1)),
                "owner":int(match.group(2)),
                "type":match.group(3),
                "randomid":match.group(4)
            }
            download_times_url="https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp"
            response_or = requests.get(url=download_times_url, params=param_or, headers=headers)
            data_download = json.loads(response_or.text)
            attach_download_times = data_download['wbshowtimes'] # 附件下载次数
            ele_data['附件下载次数'] = attach_download_times

            # 存入附件
            attach_path=os.path.join(full_path,attach_name)
            attach_response=requests.get(attach_link,headers=headers,stream=True)
            with open(attach_path,'wb+') as res:
                # 此处使用res.write(attach_response.content)疑似会卡顿，故分批
                for part in attach_response.iter_content(10240):
                    res.write(part)
            print("附件存储成功 ！ ！ ！")

        except Exception as e:
            if have_attach:
                print(f"\n【错误】在存储附件时发生异常: {type(e).__name__}")    #这个检查代码由 Gemini 提供
                print(f"【错误详情】: {e}")   #这个检查代码由 Gemini 提供
            else:
                print("无附件")
        # 记录这个通知的所有信息到列表中
        all_data.append(ele_data)

#创建 CSV文件 并且将第二三小问所提及的基础数据写入
fieldnames=['通知人','标题','日期','详情链接','附件名','附件下载次数','附件链接码']
with open("Main_Output_Data.csv",'w',encoding='utf-8',newline='') as f:
     csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
     csv_writer.writeheader()
     csv_writer.writerows(all_data)