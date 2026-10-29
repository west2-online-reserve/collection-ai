import re
password = "121212sasd121"
pattern = r"^[a-z0-9A-Z]{6,18}$"
a = re.findall(pattern,password)
if a:
    print("YES")
else:
    print("NO")