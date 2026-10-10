import re
it = re.finditer(r"\d+","和好打帝国121321313啊大大大大121")
for i in it:
    print(i.group())
