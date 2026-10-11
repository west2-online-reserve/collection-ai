def count(num):
    result={}
    for n in num:
       result[n]=result.get(n,0)+1
    return result
s=input('输入数字')
nums=[x for x in s.split()]
print(count(nums))