import numpy as np
import pandas as pd
grade = np.random.randint(0,101,(50,3))
subject = ['语文','数学','英语']
df = pd.DataFrame(grade,
                  index = [f"学生{i:02d}" for i in range(1,51)],
                  columns = subject
                  )
df['平均分'] = df[subject].mean(axis=1)
df['总分'] = df[subject].sum(axis=1)
mean_subject = df[subject].mean()
max_student = df['总分'].idxmax()

cond = (df['数学'] > 80) & (df['英语'] > 80)
df_2 = df[cond]
top_10 = df_2.sort_values('总分',ascending=False)
print(top_10)


