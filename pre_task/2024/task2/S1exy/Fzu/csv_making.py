import csv
import time



"""
封装最后的csv文件 主要对文件进行写入操作
"""

def new_csv():
    # 定义CSV文件的路径
    csv_file_path = 'output.csv'

    # 打开文件并创建CSV写入器
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["通知人", "标题", "日期","详情链接","附件下载总次数","附件名(以空格作为间隔)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入数据
        writer.writeheader()

    print(f"文件已创建 {csv_file_path}")

def write_csv(fzu_human, fzu_body, fzu_time ,fzu_header,sum,name):
    data = [[fzu_human, fzu_body, fzu_time ,fzu_header,sum,name]]

    csv_file_path = 'output.csv'
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(current_time)
        writer.writerows(data)
        print("写入成功" , current_time)

