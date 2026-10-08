s = input().split()
def sort(s):
    for i in range(len(s)-1):
        for j in range(len(s)-1):
            if s[j] < s[j+1]:
                temp = s[j]
                s[j]  = s[j+1]
                s[j+1] = temp

sort(s)    
print(s)

