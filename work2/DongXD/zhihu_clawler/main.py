"""
仅要求对一个话题进行爬取（爬取 20 条问题，每个问题爬取 10 条回答即可），学有余力的可以从话题广场开始爬。
将爬取内容存储到 CSV 文件中，格式为：问题名、问题具体内容、回答信息（只需要留下纯文字），学有余力的可以把评论也保留下来。
"""

from DrissionPage import Chromium
import pandas as pd

results_tmp=[
    {
        "问题名":"",
        "问题具体内容":"",
        "回答":[
            {
                "回答信息":""
            }
        ]
    }
]

QUESTION_NUM=10
ANSWER_NUM=20

results=[]
browser=Chromium()
tab=browser.latest_tab
tab.get("https://www.zhihu.com/topic/19554298/top-answers")

questions_ele=[]

while True:
    questions_ele=tab.eles(
    '@data-za-detail-view-element_name=Title',timeout=3)    
    if len(questions_ele)>=QUESTION_NUM : break
    tab.scroll(-1)
    tab.scroll(10000)

questions_urls=[ questions_ele[i].attr('href') for i in range(QUESTION_NUM)]

now=1

for url in questions_urls:

    print(f'{now}/{QUESTION_NUM}')

    tab.get(url)
    title=tab.ele(
        "@class=QuestionHeader-title").text

    print("正在获取问题\n",title)

    detail_ele=tab.ele(
        '@class=QuestionRichText QuestionRichText--expandable QuestionRichText--collapsed'
        ,timeout=3)
    
    if detail_ele:
        detail_ele.click()

    detail_text_ele=tab.ele(
        '@class=RichText ztext css-1olvdus'
        ,timeout=3)
    
    detail=""
    if detail_text_ele :
        detail=detail_text_ele.text

    tab.ele(
        '@class=QuestionMainAction ViewAll-QuestionMainAction'
        ,timeout=3).click()
    
    answers_ele=[]

    while True:
        answers_ele=tab.eles(
            '@class=RichText ztext CopyrightRichText-richText css-1olvdus')
        if len(answers_ele)>=ANSWER_NUM : break
        tab.scroll(-1)
        tab.scroll(10000)

    answers=[{"回答信息":answers_ele[j].text} for j in range(ANSWER_NUM)]
    results.append({
        "问题名":title,
        "问题具体内容":detail,
        "回答":answers
    })

    now+=1

print(results)
df=pd.DataFrame(results)
df.to_csv('results.csv',index=False,encoding='utf-8')


