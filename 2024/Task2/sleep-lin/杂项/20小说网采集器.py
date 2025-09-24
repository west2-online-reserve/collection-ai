import os,tqdm
import requests
from bs4 import BeautifulSoup

# 网站地址和书本ID
web = 'https://www.20xs.org'
while True:
    print("小说网站为 https://www.20xs.org ")
    input_id = input('请输入想查询的图书id(大部分为1-6位数):')
    target = web + '/' + input_id

    # 获取网页内容
    try:
        req = requests.get(url=target)
        req.raise_for_status()  # 检查请求是否成功
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e},原因可能是不在的图书id")
        exit()

    b = BeautifulSoup(req.text, 'lxml')

    # 获取书名、作者、书籍描述、章节数
    meta_tag = b.find("meta", {"property": "og:novel:book_name"})
    author = b.find('meta',{"property":"og:novel:author"})['content']
    description = b.find('div',id='intro').find('p').string
    number = len(b.find_all('div', id='list')[0].find_all('a'))
    if meta_tag:
        book_name = meta_tag['content'] 
        break
    else:
        print("图书id错误,请重新输入")


#爬取询问
print(f'书名:{book_name}\n作者:{author}\n书籍描述:{description}\n章节数:{number}')
choice = input('是否进行爬取:(y/n):')
if choice == 'y':
    # 查找章节列表
    div_tag = b.find_all('div', id='list')
    if not div_tag:
        print("无法找到章节列表")
        exit()

    a_tags = div_tag[0].find_all('a')

    # 创建文件夹
    folder_path = os.path.join('爬取的东西', '小说', book_name)
    os.makedirs(folder_path, exist_ok=True)

    # 定义文件路径
    file_path = os.path.join(folder_path, f'{book_name}.txt')

    # 将章节列表写入文件
    with open(file_path, 'w', encoding='utf-8') as fp:
        for each in a_tags:
            # 写入章节标题和链接
            chapter_url = web + each.get('href')
            fp.write(each.string + ' ' + chapter_url + '\n')

    # 逐个章节抓取内容
    with open(file_path, 'a', encoding='utf-8') as fp:
        for i, each in tqdm.tqdm(enumerate(a_tags), total=len(a_tags)):
            chapter_url = web + each.get('href')
            try:
                chapter_req = requests.get(url=chapter_url)
                chapter_req.raise_for_status()  # 检查请求是否成功
            except requests.exceptions.RequestException as e:
                print(f"获取章节 {each.string} 时出错: {e}")
                continue

            b = BeautifulSoup(chapter_req.content, 'lxml')
            div_tags = b.find_all('div', id='content')
            if not div_tags:
                print(f"未找到章节内容 {each.string}")
                continue
            
            # 获取章节段落并写入文件
            p_tags = div_tags[0].find_all('p')
            fp.write(f'\n\n章节：{each.string}\n\n')
            for p in p_tags:
                if p.string:  # 避免写入 None
                    fp.write(p.string + '\n')

    print(f"采集完成，内容已保存至文件夹：{folder_path}")
