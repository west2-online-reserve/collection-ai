import pandas as pd
import re

datasource = pd.read_csv(
    'AIsolution/work3/fzuwork/fzu.csv', header=0, encoding='utf-8')

name = datasource['标题']
ans = 0
for index, value in name.items():
    if re.match('关于', value):
        ans += 1
print(ans)
print(round(ans/name.shape[0]*100, 2))
