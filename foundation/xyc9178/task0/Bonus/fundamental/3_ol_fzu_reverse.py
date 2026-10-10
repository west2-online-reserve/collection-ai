s = input()
s = s.replace("ol","fzu")
l1 = list(s)
l2 = []
for i in range(len(s)):
    l2.append(l1[len(s)-i-1])
s = "".join(l2)
print(s)
