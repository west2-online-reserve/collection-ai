import pandas as pd


datasource = pd.read_csv(
    'AIsolution/work3/fzuwork/fzu.csv', header=0, encoding='utf-8')
datasource.dropna(axis=0,  thresh=1, inplace=True)
dic = {}
for index, row in datasource.iterrows():
    if row['通知人'] in dic:
        dic[row['通知人']] += 1
    else:
        dic[row['通知人']] = 1
# print(datasource.shape[0])
for key in dic.keys():
    dic[key] = dic[key]/datasource.shape[0]*100
    dic[key] = round(dic[key], 2)
    dic[key] = str(dic[key])+'%'
# print(dic)
ans = pd.DataFrame(dic, index=['百分比'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(ans)
# result = datasource['通知人'].value_counts()
# print(result)
