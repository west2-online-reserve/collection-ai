#输入⼀个字符串，判断字符串中是否含有 "ol" 这个⼦串
#若有把所有的 "ol" 替换为 "fzu"，最后把字符串倒序输出

s = input()
while 'ol' in s:
    s = s.replace('ol','fzu')
print(s)