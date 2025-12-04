from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).absolute().parent
pd.options.display.unicode.east_asian_width = True


#统计通知人
df = pd.read_csv(ROOT_DIR/"data.csv")
df.columns = ['notifier','title','date','url','attachments','downloads']
notifier_counts = df['notifier'].value_counts()
notifier_counts.index.name = None
notifier_percent = notifier_counts.map(lambda num : f"{(num * 100 / df.shape[0]):.2f}%")

result_notifier = pd.concat([notifier_counts,notifier_percent],axis=1)
result_notifier.columns = ['通知数','百分比']
result_notifier = result_notifier.transpose()


#统计附件下载次数
#因为之前数据输出格式的问题，这里做了一下转化
df['downloads'] = df['downloads'].map(lambda total : sum([int(num) for num in total.split(',')]) )

average_downloads = (df.groupby('notifier')['downloads'].sum() / df.groupby('notifier')['attachments'].sum())
average_downloads.index.name = None
average_downloads = average_downloads.fillna(0.00)
average_downloads = average_downloads.map(lambda num : round(num, 1))

attachment_notices = df.groupby('notifier')['attachments'].apply(lambda num : num.map(lambda x : int(x != 0)).sum())

result_downloads = pd.concat([attachment_notices,average_downloads],axis=1)
result_downloads.columns = ['含附件的通知数', '附件平均下载数']
result_downloads.sort_values(by='附件平均下载数',ascending=False,inplace=True)
result_downloads = result_downloads.transpose()


#统计日期
#学期初9、3 学期中10-12、4-5 学期末1、6 假期2、7-8
df['date'] = pd.to_datetime(df['date'])
date_frame = pd.DataFrame(columns=['year','month','day'])

date_frame['year'] = df['date'].apply(lambda date : date.year)
date_frame['month'] = df['date'].apply(lambda date: date.month)
date_frame['day'] = df['date'].apply(lambda date : date.day)

ini = ((date_frame['month'] == 9) | (date_frame['month'] == 3)).value_counts()[True]
mid = (((date_frame['month'] >= 10) & (date_frame['month'] <= 12)) | ((date_frame['month'] >= 4) & (date_frame['month'] <= 5))).value_counts()[True]
fin = ((date_frame['month'] == 1) | (date_frame['month'] == 6)).value_counts()[True]
hol = ((date_frame['month'] == 2) | ((date_frame['month'] >= 7) & (date_frame['month'] <= 8))).value_counts()[True]

result_date = pd.DataFrame(columns=['学期初','学期中','学期末','假期'])
result_date['学期初'] = [ini, f"{ini * 100 / df.shape[0]:.1f}%", round(ini / 2, 2)]
result_date['学期中'] = [mid, f"{mid * 100 / df.shape[0]:.1f}%", round(mid / 5, 2)]
result_date['学期末'] = [fin, f"{fin * 100 / df.shape[0]:.1f}%", round(fin / 2, 2)]
result_date['假期'] = [hol, f"{hol * 100 / df.shape[0]:.1f}%", round(hol / 3, 2)]
result_date.index = ['通知数','百分比','月平均']


#自由发挥环节
holiday_notices = df['title'].apply(lambda title : "放假" in title).value_counts()[True]
nice_url = df['url'].apply(lambda url : ".htm" in url).value_counts()[True]

data = {"放假通知": [f"{holiday_notices * 100 / df.shape[0]:.1f}%"],
          "优雅网址": [f"{nice_url * 100 / df.shape[0]:.1f}%"]}

result_free = pd.DataFrame(data=data,index=['百分比'])


#制作图表
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(12, 8))

#图表一
plt.subplot(2,2,1)
plt.title('各通知人占比')
plt.pie(result_notifier.loc['通知数'],labels=result_notifier.columns,autopct='%.2f%%')
plt.legend(title="通知人", loc="center left", bbox_to_anchor=(-0.5, 0.5))

#图表二
plt.subplot(2,2,2)
plt.title('每月通知数')
month_notices = date_frame['month'].value_counts().sort_index()
x = np.array(month_notices.index)
y = np.array(month_notices)
plt.bar(x,y)
plt.xticks(x)
plt.ylabel('通知数')
plt.xlabel('月份')
for x0, y0 in zip(x, y):
    plt.text(x0, y0, y0, ha='center', va='bottom', fontsize=10)

#图表三
plt.subplot(2,2,3)
plt.title('各时期月平均通知数')
x = np.array(result_date.columns)
y = np.array(result_date.loc['月平均'])
plt.yticks(range(0,300,20))
plt.ylabel('月平均通知数')
plt.xlabel('时期')
plt.bar(x, y)
for x0, y0 in zip(x, y):
    plt.text(x0, y0, y0, ha='center', va='bottom', fontsize=10)

#图表四
plt.subplot(2,2,4)
plt.title('各通知人平均附件下载数')
x = np.array(result_downloads.columns)
y = np.array(result_downloads.loc['附件平均下载数'])
plt.yticks(range(0,3000,500))
plt.ylabel('附件平均下载数')
plt.xlabel('通知人')
plt.bar(x, y)
for x0, y0 in zip(x, y):
    plt.text(x0, y0, y0, ha='center', va='bottom', fontsize=10)


if __name__ == "__main__":
    print(result_notifier,"\n")
    print(result_downloads,"\n")
    print(result_date,"\n")
    print(result_free,"\n")
    plt.show()