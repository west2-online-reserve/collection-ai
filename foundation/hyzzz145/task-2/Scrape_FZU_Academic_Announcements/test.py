# import os
# import csv
# # from scrape_copy import scatch,notice_request
# import requests

# ##进入程序文件夹
# os.chdir(os.path.dirname(os.path.abspath(__file__)))


# #打开csv，以及初始化
# f = open("content_test.csv",'a',newline="",encoding='utf-8')
# writer = csv.writer(f)

# writer.writerow(["nihao"])

# f.close()

# f = open("content_test.csv",'a',newline="",encoding='utf-8')
# writer = csv.writer(f)

# writer.writerow(["nihao"])

# f.close()
# a = [1,2,3,4,5,67]
# for i in range(5,2,-1):
#     print(i)


# def add_description(cls):
#     def c

#     return cls
# class My_Zoo():
#     def __init__(self):
#         pass
from lxml import etree
xml = """
<div class="fbnf clearfix doclist_searchbar top-filter" ms-controller="list_search">
    <div class="fbnf001 clearfix rv-pt">
        <div class="rt-pt">
            <li field="adddate"><span></span>
                <a href="jxtz/zhk.htm" title="综合科">综合科</a>
                <a href="jxtz/jxyx.htm" title="教学运行">教学运行</a>
                <a href="jxtz/jyjg.htm" title="教研教改">教研教改</a>
                <a href="jxtz/jhk.htm" title="计划科">计划科</a>
                <a href="jxtz/sjk.htm" title="实践科">实践科</a>
                <a href="jxtz/zlb.htm" title="质量办">质量办</a>
                <a href="jxtz/djzx.htm" title="电教中心">电教中心</a>
                <a href="jxtz/jcjs.htm" title="教材建设">教材建设</a>
                <a href="jxtz/tpxqglk.htm" title="铜盘校区管理科">铜盘校区管理科</a>
            </li>
        </div>
    </div>
</div>

"""

tree = etree.XML(xml)
text = tree.xpath("/div/div/div/li/a")
text2 = text[0].xpath("/div/div/div/li/a/text()")

print(text2)