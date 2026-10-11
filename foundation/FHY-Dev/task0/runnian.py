s=input().split()
b=[]
a=0
for i in range(int(s[0]),int(s[1])+1):
    if int(i)%4==0 and int(i)%100!=0 or int(i)%400==0:
        a+=1
        b.append(str(i))
print(a)
print(" ".join(b))