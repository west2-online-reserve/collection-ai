import re

# def decorator(func):
#     def wrapper():
#         print("before")
#         func()
#         print("after")
#     return wrapper

# @decorator
# def function():
#     print("func")

# function()

# re.compile(r'\w{2, 4}', re.)


# list = "覃飞，韦廉；农艳春.张秀菊;黄桂七,李杜茨:蓝十二   莫香桃"
# namelist = re.split(r'[;,，；:\.\s]\s*', list)
# print(namelist)

# list = """
#     堵木，猪猪
#     堵茨，牛牛
#     堵拐，蛙蛙
#     堵bia，鱼鱼
#     谢乜梦，***
#     卖乜梦，***
#     被骂，回家
#     供厄样，吃饭
# """
# pattern = re.compile(r"^(.*)，", re.M)
# for i in pattern.findall(list):
#     print(i)



import re

source = """
下面是这学期要学习的课程：

<a href='https://www.bilibili.com/video/av66771949/?p=1' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是牛顿第2运动定律

<a href='https://www.bilibili.com/video/av46349552/?p=125' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是毕达哥拉斯公式

<a href='https://www.bilibili.com/video/av90571967/?p=33' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是切割磁力线
"""

# 替换函数，参数是 Match对象
def subfunc(match):
    # Match对象 的 group(0) 返回的是整个匹配上的字符串。
    src = match.group(0)

    # Match对象 的 group(1) 返回的是第一个group分组的内容。
    # number = int(match.group(1)) + 6
    dest = f"/av{int(match.group(1)) + 6}/"

    print(f'{src} 替换为 {dest}')

    # 返回值就是最终替换的字符串
    return dest

newStr = re.sub(r"/av(\d+)/", subfunc , source)
print(newStr)







#我只是想要所有人能作为一个真正的人真正地活一次而已，为什么这么难