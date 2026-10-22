import re
a=input()
b=re.findall(r"ol",a)
if b:
    c=a.split("ol")
    d='fzu'.join(c)
    print(d[-1::-1])   
else:
    print(a[-1::-1])
       


    