import pandas as pd
import re

datasource = pd.read_csv(
    'AIsolution/work3/fzuwork/fzu.csv', header=0, encoding='utf-8')
datesoc = datasource['日期']
datesoc = datesoc.value_counts()
print(datesoc)
# 9 10 11 12 1
# 3 4 5 6 7
# 9 3 开学初
# 10 11 5 6 学期中
# 12 1 7 学期末
# 2 8假期
dic1 = {'09': 0, '03': 0}  # 开学初
dic2 = {'10': 0, '11': 0, '05': 0, '06': 0}  # 学期中
dic3 = {'12': 0, '01': 0, '07': 0}  # 学期末
dic4 = {'02': 0, '08': 0}  # 假期
for index, value in datesoc.items():
    index = str(index)
    month = re.search(f'-(.*?)-', index)
    if month:
        month = month.group(1)
        if month in dic1:
            dic1[month] += value
        elif month in dic2:
            dic2[month] += value
        elif month in dic3:
            dic3[month] += value
        elif month in dic4:
            dic4[month] += value
a1 = 0  # 开学初
a2 = 0  # 学期中
a3 = 0  # 学期末
a4 = 0  # 假期
for key in dic1.keys():
    a1 += dic1[key]
for key in dic2.keys():
    a2 += dic2[key]
for key in dic3.keys():
    a3 += dic3[key]
for key in dic4.keys():
    a4 += dic4[key]
print(f'开学初：{a1}，学期中：{a2}，学期末：{a3}，假期：{a4}')
