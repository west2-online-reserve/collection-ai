import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./task1.csv",encoding='utf-8')

tzr = df['通知人']
r1 = pd.DataFrame({
    '通知次数': tzr.value_counts(),
    '比例': tzr.value_counts(normalize=True)
    })

download = [c for c in df.columns if '下载次数' in c]
for c in download:
    df[c] = df[c].fillna(0)  # 填充NAN为0
df['总下载次数'] = df[download].sum(axis=1)

r2 = df.groupby('通知人')['总下载次数'].agg([('平均下载', 'mean'),('中位数', 'median')]).join(r1).sort_values(['平均下载','中位数'], ascending=False)
print(f"{r2}\n")

df['日期'] = pd.to_datetime(df['日期'])
pd.set_option('display.max_rows', None)  # 完整显示
r3 = df['日期'].value_counts().sort_index()  # 日期作为索引排序
print(f"{r3}\n")

df['标题长度'] = df['标题'].astype(str).str.len()
r4 = df.groupby('标题长度')['总下载次数'].agg([('平均下载', 'mean'),('中位数', 'median')])
print(f"{r4}\n")

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
r2['平均下载'].plot(kind='bar')
plt.title('各通知人平均下载')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(r3.index, r3.values)
plt.title('每日通知')

plt.tight_layout()
plt.show()