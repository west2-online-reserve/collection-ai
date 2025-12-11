import requests
import json
import re
import csv


headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"}


def main():
    for i in range(1,13):
        r = requests.post(url='https://summer-ospp.ac.cn/api/getProList',json=
                                {"pageNum": i, "pageSize": 50},headers=headers)
        r.encoding = 'utf-8'
        data = r.json()

        for c in data['rows']:
            p = requests.post(url='https://summer-ospp.ac.cn/api/getProDetail',json=
                            {"programId": c["programCode"],"type": "org"},headers=headers)
            detail = p.json()

            print(f"处理{c["programCode"]}")
            with open('task3.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                
                # 准备数据
                program_name = detail["programName"]
                difficulty = detail["difficulty"]
                
                # 安全处理techTag
                try:
                    tech_tag = json.loads(detail["techTag"])[0][1]
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    tech_tag = ""
                
                program_desc = remove_tags(detail["programDesc"])
                
                # 收集outputRequirement，清理换行符
                titles = []
                for o in detail["outputRequirement"]:
                    if o and o.get("title"):
                        # 替换换行符为空格
                        title = o["title"].replace('\n', ' ').replace('\r', '').strip()
                        if title:  # 只添加非空标题
                            titles.append(title)
                
                # 将所有标题合并为一个字符串，用空格分隔
                output_text = ' '.join(titles)
                
                # 写入一行数据（csv.writer会自动处理逗号和引号）
                writer.writerow([program_name, difficulty, tech_tag, program_desc, output_text])

    print("√ 已全部完成")

def remove_tags(i):
    o = re.compile("<.*?>")
    return re.sub(o,"",i)


if __name__ == "__main__": 
    main()