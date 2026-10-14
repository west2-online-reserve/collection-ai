import re

a=str(input())
if re.findall(r'^[A-Za-z0-9]{6,18}$',a):
    flag=True
else:
    flag=False
print(flag)