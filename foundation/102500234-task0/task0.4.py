s = input().split()
a = []
for i in s:
    try:
        int(i)
        a.append(i)
    except ValueError:
        pass
s = a
def sort(s):
    temp = 0
    for i in range(len(s)-1):
        for j in range(len(s)-1):
            if s[j] > s[j+1]:
                temp = s[j]
                s[j] = s[j+1]
                s[j+1] = temp
sort(s)                
                
print(s)