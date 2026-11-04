data=input().split()
nums=[]
for i in data:
    try:
        num=int(i)
        nums.append(num)
    except:
        pass
nums.sort()
print(nums)