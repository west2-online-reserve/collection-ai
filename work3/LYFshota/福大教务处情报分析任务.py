import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

if __name__ == "__main__":
    noti_data=pd.read_csv("D:\Python\west2AI\collection-ai\work2\LYFshota\福大教务处通知数据\通知数据.csv",
                                  encoding='utf-8',header=0)
    noti_attach_data=pd.read_csv("D:\Python\west2AI\collection-ai\work2\LYFshota\福大教务处通知数据\通知附件数据.csv",
                                  encoding='utf-8',header=0)
    #统计所有出现过的「通知人」，并计算他们各自发布的通知数量占总数的比例
    all_notifiers=noti_data.groupby('通知人')['标题'].count()
    all_notifiers_sum = all_notifiers.sum()
    all_notifiers = all_notifiers.sort_values(ascending=False)
    noti_num = all_notifiers.rename('通知数').to_frame()
    noti_num['占比(%)'] = (noti_num['通知数'] / all_notifiers_sum * 100).round(2)
    noti_num = noti_num.reset_index().rename(columns={'index': '通知人'})
    noti_num.index = range(1, len(noti_num) + 1)
    print(noti_num)

    #分析附件的下载次数与通知人是否存在某种联系
    all_attach_notifiers = noti_attach_data.groupby('通知人')['下载次数'].sum()
    all_downloads = noti_attach_data['下载次数'].sum()
    all_attach_notifiers = all_attach_notifiers.sort_values(ascending=False)
    attach_noti_num = all_attach_notifiers.rename('附件下载数').to_frame()
    attach_noti_num['占比(%)'] = (attach_noti_num['附件下载数'] / all_downloads * 100).round(2)
    attach_noti_num = attach_noti_num.reset_index().rename(columns={'index': '通知人'})
    attach_noti_num.index = range(1, len(attach_noti_num) + 1)
    print(attach_noti_num)
    attach_noti_num.plot.pie(y='占比(%)', labels=attach_noti_num['通知人'], autopct='%1.1f%%', title='各通知人附件下载次数占比情况')
    plt.show()

    #统计每天发布的通知数量
    daily_noti = noti_data.groupby('日期')['标题'].count()
    print(daily_noti)
    # 为多个年份绘制以月份为 X、日为 Y 的气泡图（保存每年图像）
    for year in range(2020, 2026):
        daily_noti_year = daily_noti[daily_noti.index.str.startswith(str(year))]
        if daily_noti_year.empty:
            print(f"{year} 年没有数据，跳过")
            continue

        df_year = daily_noti_year.rename('通知数').to_frame().reset_index()
        df_year['日期'] = pd.to_datetime(df_year['日期'], errors='coerce')
        df_year = df_year.dropna(subset=['日期']).sort_values('日期')
        df_year['月份'] = df_year['日期'].dt.month
        df_year['日'] = df_year['日期'].dt.day

        plt.style.use('ggplot')
        plt.figure(figsize=(12, 6))
        # 气泡大小按当日通知数归一化，然后放大到合适范围
        sizes = (df_year['通知数'] / df_year['通知数'].max()) * 1000
        sc = plt.scatter(df_year['月份'], df_year['日'], s=sizes, c=df_year['通知数'], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label='通知数')

        plt.xticks(range(1, 13), [f"{m}月" for m in range(1, 13)])
        plt.yticks(range(1, 32))
        plt.xlabel('月份')
        plt.ylabel('日')
        plt.title(f'{year}年各月每日通知数（气泡图）')
        plt.tight_layout()
        plt.show()

