def number(nums):
    d={}
    for num in nums:
        if num in d:
            d[num]=d[num]+1
        else:
            d[num]=1
    return d
s=input()
nums=list(map(int, s.split()))
result=number(nums)
print(result)
