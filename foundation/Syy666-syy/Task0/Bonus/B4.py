s=input('输入列表')
it=s.split()
numbers=[]
for i in it:
    try:
        numbers.append(int(i))
    except ValueError:
        pass
numbers.sort()
print(numbers)