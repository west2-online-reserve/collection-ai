# task-2 的学习记录

## 9 月 29 号

- 准备安装 numpy 。但发现电脑没有 pip 模块，无法安装 pip，所以我准备借鉴 deepseek 说的安装虚拟环境。
- 我做了以下操作
  - 首先，cd 到 task-2。
  - python -m venv myenv，在 task-2 目录里生成了 myenv
  - .\myenv\bin\Activate.ps1 激活
  - 哎，卡安装这步了

## 9 月 30 日

- 可能是我安装的 python 太新了，没有对应的 wheel，我现在尝试使用其他版本的 python
- ok，换了 3.13 的 python 就可以安装 2.5.3 的 numpy了，现在激活的命令为
  - .\myenv\Scripts\Activate.ps1
  - deactivate

## 10 月 1 日

- 学习完 numpy 基本知识

## 10 月 2 日

- 安装 3.0.6 的 pandas
- 学习了 pandas 基础，现在开始安装 matplotlib
- 安装了 3.11.2 的 matplotlib
- 懂了些 matplotlib
- 我尝试去刷下题
- deepseek 出的题太难了 TAT
- 去学一下爬虫

## 10 月 4 日

- 网络爬虫有点不太懂，我使用 with open("mybaidu.html",'w') as f: f.write(resp.read()) 报错， chatgpt 说要用 'wb'，但改了之后，虽然成功了，但没文件出现;设置我复制 b站 老师的代码也是报错。
- 原来是我傻了，我终端地址在根目录，文件保存也在根目录

## 10 月 5 日

- 学习一下 os
  - 学习了 getcwd 和 chdir，还有就是在字符串双引号前面加 r 是忽略字符串中的 \

## 10 月 6 日

- 学习了 listdir()，用于获取当前目录文件；
学习了 mkdir，rmdir，rename，getenv
- 现在继续学习爬虫吧
  - 安装了 2.34.2 的 requests 库。学习了 .get()，还有 .post()
  - 尝试爬取 XHR，在复制 param 信息时，浏览器把 “:” 显示为 %3A ，导致我第一次运行失败。还有就是请求完记得关闭

## 10 月 7 日

- 复习一下正则表达式
  - 元字符是具有固定含义的特殊符号：. 匹配除换行符外的任意字符；\w 匹配字母、数字或下划线，\W 匹配它们以外的字符；\d 匹配数字，\D 匹配非数字；\s 匹配空白字符，\S 匹配非空白字符；\n 匹配换行符，\t 匹配制表符；^ 匹配字符串的开头，$ 匹配字符串的结尾；a|b 表示匹配 a 或 b；() 用于分组，也可以匹配括号内的表达式；[...] 匹配字符组中的任意一个字符，[^...] 匹配不在字符组中的任意一个字符。（made by chatgpt）
  - 量词用于指定它前面的字符或表达式重复的次数：* 表示重复 0 次或多次，+ 表示重复 1 次或多次，? 表示重复 0 次或 1 次，{n} 表示恰好重复 n 次，{n,} 表示至少重复 n 次，{n,m} 表示重复 n 到 m 次；例如 a{2,4} 可以匹配 aa、aaa 或 aaaa。（made by chatgpt）

## 10 月 8 日

- 复习一下正则表达式
  - 贪婪匹配 .*；惰性匹配 .*?
  - 易错，匹配 .com 要转义 \.
- re 模块
  - .findall 寻找匹配字符，并变为列表
  - .finditer 为变为迭代器，想要内容输出正常加 group()
  - .serch 只找一个结果,获取内容用 .group()
  - .match 从头匹配
  - obj = re.compile(r"\d+") 为预加载正则表达式，可以在 obj 后使用以上方法
  - re.S 是让 . 也匹配换行符
  - (?P< name>....)"可以用于特定内容的筛选
  - str.strip() 是去除前方空格
  - iter.groupdict 是把迭代器变为字典
- csv 库
  - csv.writer() 返回的是一个 csv 写入对象。
  - writerow(list) 是写入一行
- 福大教务处
  - 当我在爬取福大网页时，获得的 text 是乱码，不是保存的问题，是福大的 encoding 为 ISO-8859-1，然后我让 resp.encoding = resp.apparent_encoding 解决问题
  - 说个难崩的，我没好好审题，把【通知人】当成署名，折磨了我好久
  - 如果在使用 writerrows 时不想空行，可以在 open 里加个 newline=""
  - 在判断 finditer 类型是否有值时，可以使用 if next(finditer,None) == None: ，这样不会破坏迭代器的顺序
  - 福大爬虫非附加题部分已完成

## 10 月 9 日

- 学习 beautifulsoup
  - 第一步需要 soup = beautifulsoup(html,'lxml')
  - 然后可以对 soup 使用 .find .findall 等方法，findall(标签,属性)，对于这些方法的返回变量可以使用 object['...']查看并列的其他属性，get() 也如此，如 herf value。也可以用.get_text() 方法获得文本。还有 .parent 和 .children 属性
- 完成了福大教务处正文的提取
- __file__ 是获取当前程序的路径
- os
  - path 模块
    - dirname 获取父级
    - abspath 获取绝对路径
- 学习了 try 和 except 处理报错问题
- request 尽量加个 timeout 参数
- html
  - 标签
    - img
    - h
    - a
  - 属性
    - align
    - scr
- request
  - 请求的属性 content 代表字节，一般用于抓取图片
- 迭代器使用 itertool 的 islice 做切片s
- 福大教务处爬虫 bonus 完成
- 复习 @decorate

## 10 月 10 日

- 学习 python 小技巧
  - 字符串格式化

  ```python
  "aaa" + str
  #=
  "aaa%s" % (str)
  ```

  - s = "pass" if score > 60 else "fail"
- 学习 xpath
  - lxml
    - etree
      - parse
        - 打开文件，先识别内容类型，在解析
      - .XML(xml文件)
        - xpath
          - /../../text()拿文本，返回列表
          - //代表后代
          - /*/代表任意
          - /li[1] 指第一个 li ，并非和 path 传统从 0 开始一样
          - /a [@herf = '...']，@ 后面跟着属性
          - /a/@herf 拿属性的值Grab
          - xpath 返回后的列表中的元素也可以使用 xpath，输入为"./..."
        - etree.tostring()- 把解析后的树转换成 HTML 或 XML 内容
          - tree：要转换的文档树
          - pretty_print=True：添加换行和缩进，便于阅读
          - encoding=resp.encoding or "utf-8"：指定编码；resp.encoding 为空时使用 UTF-8
          - method="html"：按 HTML 规则输出
          - 默认返回 bytes（字节）
          - 用二进制模式写入文件：open("test.html", "wb")
          - encoding="unicode"：返回 Python 字符串 str
          - 用文本模式写入文件：open("test.html", "w", encoding="utf-8")
