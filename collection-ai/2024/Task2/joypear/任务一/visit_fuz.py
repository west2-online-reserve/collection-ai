from xpath_try import trend
import csv
with open("./title.csv", mode='w', newline='', encoding='utf-8') as file:
    header = ['日期','通知人','标题','链接']
    write = csv.writer(file)
    write.writerow(header)
trend("https://jwch.fzu.edu.cn/jxtz.htm")
for i in range(196,166,-1):
    trend(f"https://jwch.fzu.edu.cn/jxtz/{i}.htm")
    print(197-i)