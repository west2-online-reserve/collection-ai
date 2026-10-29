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
  - 学习了 getcwd 和 chdir，还有就是在字符串双引号前面加 r 是忽略字符串中的 \\

## 10 月 6 日

- 学习了 listdir()，用于获取当前目录文件；
学习了 mkdir，rmdir，rename，getenv
- 现在继续学习爬虫吧
  - 安装了 2.34.2 的 requests 库。学习了 .get()，还有 .post()
  - 尝试爬取 XHR，在复制 param 信息时，浏览器把 “:” 显示为 %3A ，导致我第一次运行失败。还有就是请求完记得关闭

## 10 月 7 日

- 复习一下正则表达式
  - 元字符是具有固定含义的特殊符号：. 匹配除换行符外的任意字符；\\w 匹配字母、数字或下划线，\\W 匹配它们以外的字符；\\d 匹配数字，\\D 匹配非数字；\\s 匹配空白字符，\\S 匹配非空白字符；\\n 匹配换行符，\\t 匹配制表符；^ 匹配字符串的开头，$ 匹配字符串的结尾；a|b 表示匹配 a 或 b；() 用于分组，也可以匹配括号内的表达式；[...] 匹配字符组中的任意一个字符，[^...] 匹配不在字符组中的任意一个字符。（made by chatgpt）
  - 量词用于指定它前面的字符或表达式重复的次数：* 表示重复 0 次或多次，+ 表示重复 1 次或多次，? 表示重复 0 次或 1 次，{n} 表示恰好重复 n 次，{n,} 表示至少重复 n 次，{n,m} 表示重复 n 到 m 次；例如 a{2,4} 可以匹配 aa、aaa 或 aaaa。（made by chatgpt）

## 10 月 8 日

- 复习一下正则表达式
  - 贪婪匹配 .*；惰性匹配 .*?
  - 易错，匹配 .com 要转义 \\.
- re 模块
  - .findall 寻找匹配字符，并变为列表
  - .finditer 为变为迭代器，想要内容输出正常加 group()
  - .serch 只找一个结果,获取内容用 .group()
  - .match 从头匹配
  - obj = re.compile(r"\\d+") 为预加载正则表达式，可以在 obj 后使用以上方法
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

## 10 月 11 日

- 知乎话题抓取
  - 直接请求，放回 403 状态码，即禁止访问
  - 我得去学习 `selenium`
- `selenium` （模拟用户上网行为）
  - `webdriver`
    - 浏览器选择方法；如（`Edge(service,option)`）
      - `get(url)`；输入网址，打开特定网页
      - `find_element(By.属性,value:"值")`
        - `send_keys(str)` 发送内容
        - `click()`：点击
        - `txet`：读取内容
        - `get_attribute`：获得标签属性
      - `save_screenshot('herf')`，截取当前图片，但不会自动创建文件夹
      - `quit()`：退出页面
      - `page_source`：获得 html
      - `service`：
      - `close()`:关闭标签页
      - `maximize_window()`：最大化
      - `minimize_window()`：最小化
      - `set_window_size(int,int)`：自定义大小
      - `fullscreen_window()`：全屏
      - `back()`：退后
      - `forward()`：向前
      - `refresh()`：刷新
      - `title`：获得标题
      - `current_url`：获得链接
      - `execute_script("arguments[0].scrollIntoView({block: 'center'});",element)`：一般用于互动到某个页面
    - `support`
      - `ui.WebDriveWait(drive,time)`：等待加载,最多等待 time 秒
        - `until(EC)`
      - `expected_conditions`
        - `visibility_of_element_located((By., ''))`：内容存在且可见
        - `presence_of_all_elements_located()`：等待页面 DOM 中出现所有匹配该定位条件的元素
    - `common`
      - `by.By`；用于上方属性，属性记得大写
        - xpath
        - 属性
        - css
      - `keys.Keys.RETURN`：回车键
- 学习一下 css
  - `#title` 表示对 `id` 为 “title” 的起效
  - `.title` 表示对 `class` 为“title” 的起效

## 10 月 12 日

- 会话
  - `request.session`
    - `post(url,data)`：其中 data 为一个字典
      - `.cookies`
  - 获得 cookies 之后再进行请求
- head
  - referer 防盗链
- `selenium.webdrive.Edge()`
  - `add_cookie(cookie)`添加cookie
  - `get_cookie()`：获得列表，列表元素为 cookie 的字典
- `json`
  - `dump(json,f)`：写入文件
  - `dumps`：写入字符串
    - 写入通用参数，`ensure_ascii=False`：中文正常显示，否则输出`\\uXXX`
    - `indent = 2`：缩进美化
  - `load`：从文件读取对象
  - `loads`：从字符串读取对象
- 学习了 `breakpoint()` 调试
  - `p x`：打印变量 x
  - `l`：显示当前代码上部
  - `n`：进行下一行代码
  - `q`：退出调试

## 10 月 13 日

- 知乎话题爬取
  - 哎，在使用 selenium get_element 时，使用 xpath 进行抓取，总是要看运气。·

## 10 月 14 日

- 知乎话题爬取
  - bug 还是太多了，我决定还是先让 chatgpt 生成一个无 bug 的版本，如何纠错
  - 哎就这样吧，搞半天只能爬取一个话题。明天去学点编程思想吧

## 10 月 15 日

- 开始源之夏
- `repr()`是一种函数，用于把对象返回为开发者表示形式
- `traceback`是一个内置模块
  - `print_exc`会把当前正在处理的异常打印下来
- 有时候爬取 json 的时候，请求失败要记得看看写没写 padload。
- 排查问题的代码

  ``` python
  print("状态码：", resp.status_code)
  print("请求方法：", resp.request.method)
  print("请求网址：", resp.request.url)
  print("请求体：", resp.request.body)
  print("响应类型：", resp.headers.get("Content-Type"))
  print("响应正文：", repr(resp.text[:1500]))
  ```

- re
  - `sub(re,"",text)`：替换text的东西
    - 例子：要去除 html 的<>类标签，使用 re.sub(r'<[^>]*>','',text)
  - "^" 在 [] 里指非的意思
- `for a,b in dict.items()`：可以使 a 为键，b 为值
- 已完成源之夏

## 10 月 16 日

- `urllib.parse`
  - `unquote(text)`：可以用于解码

## 10 月 17 日

- 语法报错
  - `SyntaxError`：无效语法，不符合语法格式。
    - 关键字命名
    - 错误符号
    - 语法不完整：符号不成对
  - `NameError`：变量名错误
  - `ValueError`：传入值错误
  - `IndentationError`：缩进错误
  - `ZeroDivisonError`：除数是 0
  - `IndexError`：索引错误
  - `FileNotFoundError`：未找到文件
  - `AttributeError`：属性错误
  - `UnboundLocalError`：引用未定义的局部变量
  - `KeyError`：未找到对应的键值
  - `TypeError`：无效类型
  - `OverflowError`：数值运算超出最大限制
  - `NotImplementedError`：尚未实现的方法
  - `ModuleNotFoundError`：模块未找到
  - `try...except`：异常处理
  - `assert`：断言
- 在请求一个 url 时，问号后面的可以用 param 字典参数来代替
- numpy（*made by chatgpt*）
  - `array(list)`：把列表转为数组
    - 数组有 `np.int32` `np.int64` `np.float32` `np.float64` `np.bool_` `np.str_`
    - `astype`：把数组类型转换
    - `reshape(1，4)`：把有4  个元素数组改为（1，4）的形式，参数也可以填 -1 自动计算
  - `zeros(n)`：创造 n 个 0 的数组，其中类型为浮点数。n 可以是 (2,3)，也可以是一个数
  - `arange(stop)` / `arange(start, stop, step)`：按步长生成连续数值，包含 `start`，不包含 `stop`。
    - `np.arange(10)`：生成 `0` 到 `9`。
    - `np.arange(1, 10, 0.5)`：生成 `1` 到 `9.5`，步长为 `0.5`。
  - `random`
    - `random(size)`：生成 `[0, 1)` 范围内的随机小数。
      - `np.random.random(5)`：生成 5 个随机数。
      - `10 * np.random.random((10, 5)) + 10`：生成形状为 `(10, 5)`、范围约为 `[10, 20)` 的随机数。
    - `randint(low, high, size)`：生成随机整数，包含 `low`，不包含 `high`。
      - `np.random.randint(2, 5, (3, 5))`：生成形状为 `(3, 5)`、范围为 `2` 到 `4` 的整数。
    - `randn(2, 3)`：生成形状为 `(2, 3)`、标准正态分布的随机数。
    - `normal(loc, scale, size)`：按指定均值、标准差和形状生成正态分布的随机数。
      - `np.random.normal(0, 1, (2, 3))`：均值为 `0`、标准差为 `1` 的随机数组。
      - `np.random.normal(500, 70, (3, 7))`：生成形状为 `(3, 7)` 的随机数组。
  - `索引与切片`
    - `arr[indices]`：按索引取值，顺序与索引列表一致
      - `np.arange(0, 100, 10)[[0, 2, 3, 1]]`：结果为 `[0, 20, 30, 10]`
    - `arr[start:stop]`：包含起点、不包含终点；二维数组默认切行
      - `arr[:, 1:3]`：取所有行的第 2、3 列
    - `arr[condition]`：按条件筛选元素
      - `arr[arr > 4]`：筛选大于 `4` 的元素
  - `转置与复制`
    - `arr.T`：转置二维数组；一维数组转置后仍是一维
    - `copy()`：创建独立副本；直接赋值会让变量共享同一个数组
      - `part = arr[:3].copy()`：修改 `part` 不影响原数组
      - `arr2 = arr`：修改 `arr2` 会影响 `arr`
  - `翻转`
    - `fliplr(arr)`：左右翻转二维数组
    - `flipud(arr)`：上下翻转；一维数组也可使用
  - `拼接与分割`
  - `concatenate(arrays, axis)`：拼接数组；`axis=0` 按行，`axis=1` 按列
  - `split(arr, indices, axis)`：按指定索引位置分割数组
    - `np.split(np.arange(10, 100, 10), [2, 8])`：按索引 `2`、`8` 分成三段
  - `数学与统计`
    - `dot(a, b)`：计算点积
      - `np.dot(np.arange(5), np.arange(5))`：结果为 `30`
    - `abs(arr)`、`sin(arr)`、`exp(arr)`、`log(arr)`：绝对值、正弦、指数、自然对数
    - `pi`：圆周率
    - `max`、`sum`、`prod`、`mean`、`std`：最大值、求和、求积、均值、标准差
      - `axis=0`：按列计算；`axis=1`：按行计算
      - `np.max(np.arange(10).reshape(2, 5), axis=0)`：结果为 `[5, 6, 7, 8, 9]`
      - 函数名前加 `nan` 可忽略 NaN，如 `np.nanmean(arr)`、`np.nansum(arr)`
  - `条件定位`
    - `where(condition)`：返回满足条件的位置索引；二维数组返回行、列索引
      - `np.where(arr > 600)`：返回大于 `600` 的元素所在位置

## 10 月 18 日

- pandas（*made by chatgpt*）
  - `Series`
    - `Series(data, index)`：创建带索引的一维数据
      - `pd.Series({"a": 1, "b": 2})`：字典键作为索引，值作为数据
      - `pd.Series([1, 2, 3], index=["a", "b", "c"])`：用 `index` 指定索引
    - `sr.values`：获取底层数组；通常是 NumPy 数组
  - `DataFrame`
    - `DataFrame(data, index, columns)`：创建二维表
      - `pd.DataFrame({"年龄": [12, 35], "性别": ["女", "男"]})`：用字典按列创建
      - `pd.DataFrame({"年龄": p1, "性别": p2})`：用 Series 建表，按索引对齐数据
      - `df.T`：行列转置
    - 二维数组含数字和字符串时，数组可能会把所有值统一成字符串；按列创建更容易保留类型
  - 选择与修改
    - `loc[行标签, 列标签]`：按标签选择
      - `df.loc["1号"]`：选一行
      - `df.loc[["1号", "2号"]]`：选多行
      - `df.loc["2号", "年龄"] = 1`：修改指定单元格
    - `iloc[行位置, 列位置]`：按整数位置选择
      - `df.iloc[1]`：选第 2 行
      - `df.iloc[:]`：选择所有行
    - `iloc[行切片, 列切片]`：按位置切片或翻转
      - `df.iloc[:, ::-1]`：反转列顺序
      - `df.iloc[::-1, :]`：反转行顺序
  - 读取与概览
    - `read_csv(path, index_col=0)`：读取 CSV；将第 1 列设为索引
      - `pd.read_csv("data.csv", index_col=0)`：行索引是第一列
    - `head(n)`：查看前 `n` 行，默认 5 行
    - `describe()`：查看数值列的描述性统计，文本列自动排除，除非加 `include = 'all'`
  - 缺失值
    - `isna()` / `isnull()`：检测缺失值
    - `dropna(axis=0)`：删除含缺失值的行；`axis=1` 删除含缺失值的列
    - `fillna(value)`：填充缺失值
      - `df.fillna(0)`：用 `0` 填充
      - `df.ffill()`：用前一个值填充
      - `df.bfill()`：用后一个值填充
  - 分组与透视表
    - `pivot_table(values, index, aggfunc)`：按行索引分组并汇总，默认计算均值
      - `df.pivot_table("是否生还", index="性别")`：按性别统计平均生还值
  - 连续数据分箱
    - `cut(values, bins)`：按指定区间分箱，默认区间右侧闭合
      - `pd.cut(df["年龄"], [0, 25, 120])`：分为 `(0, 25]` 和 `(25, 120]`
    - `qcut(values, q)`：按分位数分箱，各箱数据量大致相等
      - `pd.qcut(df["费用"], 2)`：按中位数分为两组

## 10 月 19 日

- `pandas`
  - `df`
    - `df.values`可以把数据转为二维数组
    - `df.set_index('')`设置新索引
    - `df[].rank(ascemding,method)`返回从大到小的位次 series，且同值同排名，且类型为 float64
    - `index(index)`获得索引名称
    - `values(index)`获得值
  - `series`
    - `.astype()`转类型
- `numpy`
  - `sum(array,axis=)`axis 为 1 是横着加；为 0 是竖着加
  - `argsort(array)`把数组从小到大标号返回索引数组
  - `empty(len,dtype)`创造一个 len 长度的 dtype 类型空数组
  - `argmax(array)`最大值的索引
- 安装了 Jupyter Notebook
  - 启动 `jupyter notebook`
  - Shift + Enter 运行单元格
  - Esc + A 上方插入新单元格
  - Esc + B 下方插入新单元格
  - Esc + M 切换为 Markdown 格式
  - Ctrl + Enter 运行当前单元格
  - shift + enter 运行单元格并跳转下一个
- itertool
  - `pairwise(iter)`当使用 for 循环时，返回类似 （1，2）（2，3）（3，4）...
- 在数据分析发现的几个问题
  - 写入 csv 时，输入''空字符会在csv中带引号
  - writerrows 要写二维数组

## 10 月 20 日

- `itertools`
  - `chain(list,iter)`：把list里的元素塞到iter头部
  
## 10 月 21 日

- `pd.dataframe().set_index('',**inplace**)`：一般来说，如果不加 inplace，只支持放回修改的新 df，加了后，直接对原 df 进行修改
- `pd.df.to_csv('name.csv',float_format='%.2f')`：即可导入csv,float_format 可以设置保留几位小数
- 在 `pd.df`里，如果已经`set_index`了，不能直接 df[索引列名]

## 10 月 22 日

- 如果在使用 `pd.df.to_csv` 时发现有空行现象，记得在 str 数据里加 `.stripe()` 方法
- `pd.df.to_csv` 的 `mode = 'a'` 表示写入模式
- 正则表达式可以 `search('()',txet).group(1)`找到匹配对象
- 准备尝试用英语写注释
- 算了，拿英语写太累了
- `matplotlib.pyplot`(哎，这种笔记还是得自己写)
  - 无法使用中文打以下代码

  ```python
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
  ```

  - `figure(figsize(n*n),dpi=)`创建图像,`figsize`是图片大小，`dpi`是清晰度
  - `title('',fontszie='')`设置标题
  - `plot(x,y)`：其中 x 和 y 是数组或列表，x y 一一对应、
  - `axes()`：用于 oop 的建造图表
    - `plot(x,y)`；同上
  - `show()`：显示
  - `subplot(3,1,1)`：建 3 行 1 列的图表，使用 plot 在第一个图绘制
    - 若使用 `axes` 代码如下

    ```python
    # # 面向对象方式
    # Fig2, ax2 = plt.subplots(3)     # 创建一个包含3个子图的画布，ax2 是子图数组
    # ax2[0].plot(x,y1)               # 在第1个子图上绘制 x-y1 折线图
    # ax2[1].plot(x,y2)               # 在第2个子图上绘制 x-y2 折线图
    # ax2[2].plot(x,y3)               # 在第3个子图上绘制 x-y3 折线图
    # x = np.linspace(0, 10, 50)
    # y = np.sin(x)
    ```
  
  - 图分类
    - `plot(x,y)`：线型图
    - `pie(value,lable,autopct=lambda p:express)`：饼图，若 `autopct='%1.2f%%'` 则表示显示一个浮点数，保留 2 位小数，字宽1；
    - `scatter(x,y)`：散点图
    - `bar(x,y)`：条形图
    - `stem(x,y)`：杆图
    - `step(x,y)`：阶梯图
  - `bar_label(bar,padding=3,fmt='%.2f')`对柱形图上显示数字，`padding`代表距顶高度
- `np`
  - `linespace(0,10,50)`：在 0，10 上均匀分布50 个数
- `{1.100:g}` 表示抹去多余的 0，返回 1.1

## 10 月 23 日

- `numpy`
  - `array`
    - `reshape(a,b)`返回一个 a 行 b 列的矩阵
    - `astype(type)`返回一个转类型的数组
    - `T`转置二维数组；一维数组转置后仍是一维
    - `fliplr(arr)`：左右翻转二维数组
    - `flipud(arr)`：上下翻转；一维数组也可使用
    - `copy()`：创建独立副本；直接赋值会让变量共享同一个数组
    - `size`获得 size
    - `transpose(1,0,2)`让原来的维度重新排列
  - `zero(n)`：创造 n 个 0 的数组，其中类型为浮点数。n 可以是 (2,3)，也可以是一个数
  - `arange(stop)` / `arange(start, stop, step)`：按步长生成连续数值，包含 `start`，不包含 `stop`。
  - `random`
    - `random(size)`生成 `[0, 1)` 范围内的随机小数。
    - `randint(low,high,size)`生成随机整数，包含 `low`，不包含 `high`。
    - `randn(size)`均值为0 标准差为 1 的正态分布
    - `normal(average,标准差,size)`状态发布
  - `concatenate((a, b), axis=0)`拼贴两个数组，axis=0 是按照上下拼贴，axis = 1 是左右拼贴
  - `split(array,[a,b....])`切割数组，返回切割后的列表
  - `abs(array)`
  - `sin(array)`
  - `exp(array)`
  - `log(array)`
  - `max`、`sum`、`prod`、`mean`、`std`：最大值、求和、求积、均值、标准差
  - `argmax(array)`：返回数组最大值的索引，不过是展开后一维的
  - `unravel_index(idx,size)`针对一位索引，转为二维索引(row,column)
  - `newaxis` 用来给数组增加一个长度为 1 的维度，常用于把一维数组变成行向量或列向量
  - `any(array condition,axis)`返回一个降维 bool 数组来表示符合条件的部分，这里的 aixs 指沿着某个维度
  - `where(bool 掩码)`返回索引
  - `stack([array1,array2],axis=)`增维拼贴
  - `repeat(array,n,axis)`返回把 axis 维重复 n 次
  - `clip(a,min,max)`把数组 a 限制在  min 到 max
  - `swapaxes(array,0,1)`返回的让 1 维和 2 维换位置的数组
