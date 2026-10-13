list1 = []
x = int(input())
y = int(input())
z = int(input())
list1.append(x)
list1.append(y)
list1.append(z)
list1.sort(reverse=True)
print( list1[0], list1[1], list1[2])