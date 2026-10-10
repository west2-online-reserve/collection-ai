import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

data = {
    '订单号': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08'],
    '城市': ['北京', '上海', np.nan, '北京', '广州', '上海', '深圳', np.nan],
    '销售额': [1200, 800, 1500, np.nan, 950, 2000, 1100, 1300],
    '数量': [2, 1, 3, 2, np.nan, 4, 2, 3]
}

df = pd.DataFrame(data)


df['城市'] = df['城市'].fillna(df['城市'].mode()[0])
df['销售额'] = df['销售额'].fillna(df['销售额'].mean())
df['数量'] = df['数量'].fillna(df['数量'].median())  
df['单价'] = df['销售额']/df['数量']

result = df.groupby('城市').agg(
    总售价 = ('销售额','sum'),
    平均单价 = ('单价','mean'),
    订单数 = ('数量','sum')
    
)


max_city = result['总售价'].idxmax()

fig,axes = plt.subplots(2,2,figsize=(12, 8))
axes[0,0].bar(result.index,result['总售价'])
axes[0,0].set_title('总销售额')
axes[0,0].set_xlabel('城市')
axes[0,0].set_ylabel('销售')

axes[0,1].pie(result['订单数'],labels = result.index,autopct='%1.1f%%')

axes[1,0].plot(df['订单号'],df['销售额'])
axes[1, 0].set_title('订单销售额变化')
axes[1, 0].set_xlabel('订单号')
axes[1, 0].set_ylabel('销售额')

for city, g in df.groupby('城市'):
    axes[1, 1].scatter(g['数量'], g['销售额'], label=city)
axes[1, 1].set_title('数量 vs 销售额')
axes[1, 1].set_xlabel('数量')
axes[1, 1].set_ylabel('销售额')
axes[1, 1].legend()


plt.show()

