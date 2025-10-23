import pandas as pd

datasource = pd.read_csv(
    'AIsolution/work3/fzuwork/fzu.csv', header=0, encoding='utf-8')

datasource.dropna(axis=0, thresh=6, inplace=True)
dic = {}
for index, row in datasource.iterrows():
    if row['通知人'] not in dic:
        dic[row['通知人']] = [1, row['附件下载次数']]
    else:
        dic[row['通知人']][0] += 1
        dic[row['通知人']][1] += row['附件下载次数']

for key in dic.keys():
    dic[key][1] /= dic[key][0]
    dic[key][1] = round(dic[key][1], 0)
ans = pd.DataFrame(dic, index=['有附件的通知数', '平均附件下载次数'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
ans.sort_values(by='平均附件下载次数', axis=1, ascending=False, inplace=True)
print(ans)
