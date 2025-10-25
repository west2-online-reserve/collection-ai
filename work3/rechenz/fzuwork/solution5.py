import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasource = pd.read_csv(
    'AIsolution/work3/fzuwork/fzu.csv', header=0, encoding='utf-8')
datasource['日期'] = datasource['日期'].str[6:11]
datesoc = datasource['日期']
# print(datesoc)
datesoc = datesoc.value_counts()
datesoc.sort_index(inplace=True)
# print(datesoc)
x = np.array(datesoc.index)
y = np.array(datesoc)
plt.figure(figsize=(100, 50))
plt.plot(x, y)
plt.show()
