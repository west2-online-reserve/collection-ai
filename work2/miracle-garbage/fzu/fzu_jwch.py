
import requests
from lxml import etree
import re
import csv

if __name__=='__main__':
    url_front='https://jwch.fzu.edu.cn/jxtz/'
    url_end='.htm'
    
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/146.0'
    }
    
    author_list=[]
    title_list=[]
    date_list=[]
    detail_url_list=[]

    add_list_url=[]
    add_list_name=[]
    add_list_times=[]

    for page_num in range(167,207): # page_num 指的是url参数上的页数表示
        # 确定域名参数
        url=url_front+str(page_num)+url_end

        # 发起请求获取页面源代码
        response = requests.get(url=url,headers=headers)
        text=response.content.decode('utf-8',errors='ignore')
    
        # 预处理用于搜索节点 
        html=etree.HTML(text)

        # 标题
        title_list_tmp=html.xpath(r'//ul[@class="list-gl"]/li/a/@title')
        for i in title_list_tmp:
            title_list.append(i)
        
        # 发起人
        author_list_tmp=html.xpath(r'//ul[@class="list-gl"]/li/text()')
        for i in author_list_tmp:
            if i.strip():
                author_list.append(i.strip())
    
        # 日期
        date_list_tmp=html.xpath(r'//ul[@class="list-gl"]/li/span/text()')
        for i in date_list_tmp:
            date_list.append(i.strip())

        # url
        current_url_list=[]
        detail_url_list_tmp=html.xpath(r'//ul[@class="list-gl"]/li/a/@href')
        for i in detail_url_list_tmp:
            i='https://jwch.fzu.edu.cn'+i.strip('.')
            detail_url_list.append(i)
            current_url_list.append(i)

        # 详情页面
        cnt=0
    
        for url_detailed in current_url_list:
            response_detail=requests.get(url=url_detailed,headers=headers)
            html_text=response_detail.content.decode('utf-8')

            # 判断附件
            html_xpath=etree.HTML(html_text)
            add_list=html_xpath.xpath(r'//ul[@style="list-style-type:none;"]/li/a/@href')
            if add_list:
                # 附件链接
                add_list_url.append(add_list)

                # 附件名称
                text_add=html_xpath.xpath(r'//ul[@style="list-style-type:none;"]/li/a/text()')
                name_list=[]
                for i in text_add:
                    name_list.append(i.strip())
                add_list_name.append(name_list)

                # 附件下载次数
                click_times_list=[]
                for i in add_list:
                    # 提取参数
                    owner=re.search(r'owner=(\d+)',i).group(1)
                    wbfileid=re.search(r'wbfileid=(\d+)',i).group(1)
                    param={
                        'wbnewsid':wbfileid,
                        'owner':owner,
                        'type':'wbnewsfile',
                        'randomid':'nattach'
                    }
                    response_click=requests.get(url='https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp',headers=headers,params=param)
                    click_times=re.search(r'"wbshowtimes":(\d+)',response_click.text).group(1)
                    click_times_list.append(click_times)
                add_list_times.append(click_times_list)
                    
            else:
                add_list_url.append('')
                add_list_name.append('')
                add_list_times.append('')

            cnt+=1
            print(page_num,'页',cnt,'条')
            
        print(page_num)

    # 输出
    info_list=[]

    for author,title,date,url,add_name,add_times,add_url in zip(author_list,title_list,date_list,detail_url_list,add_list_name,add_list_times,add_list_url):
        dict_info={
            '科室':author,
            '标题':title,
            '发布日期':date,
            '详情链接':url,
            '附件名':add_name,
            '附件下载次数':add_times,
            '附件链接':add_url
        }
        info_list.append(dict_info)
        print(author,title,date,sep='')


    # 定义CSV表头（与字典的key对应）
    headers = ['科室', '标题', '发布日期','详情链接','附件名','附件下载次数','附件链接']
    
    # 写入CSV（encoding='utf-8-sig' 解决Excel打开中文乱码）
    
    from pathlib import Path
    # 一行获取脚本所在文件夹（Path 对象，可直接拼接）
    current_script_dir = Path(__file__).parent

    # 直接拼接文件路径（无需手动加分隔符）
    file_path = current_script_dir / "info.csv"  # / 是路径拼接符
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        # DictWriter按表头写入，自动对齐列
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()  # 写入表头
        writer.writerows(info_list)  # 写入所有数据行




    
    