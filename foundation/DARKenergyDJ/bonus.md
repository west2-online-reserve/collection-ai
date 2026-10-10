# ***bonus***

## xyz

### planA

```python
l0 = []
for i in range(3):
    l0.append(input("三个不同的数："))
x = l0[0]
y = l0[1]
z = l0[2]
p1 = max(int(x),int(y),int(z))
p3 = min(int(x),int(y),int(z))
l0.remove(str(p1))
l0.remove(str(p3))
p2 = l0[0]
print(p1,">",p2,">",p3)
```

### planB

```python
x = int(input())
y = int(input())
z = int(input())
max = x if x > y and x > z else y if y > z else z
min = x if x < y and x < z else y if y < z else z
mid = x if x > min and x < max else y if y > min and y < max else z
print("{}>{}>{}".format(max,mid,min))
```

### planC

```python
l0 = []
for i in range(3):
    l0.append(int(input()))
m1 = [l0[i] for i in range(3) if l0[i] == max(l0)]
m2 = [l0[i] for i in range(3) if l0[i] == min(l0)]
m3 = [l0[i] for i in range(3) if l0[i] != max(l0) and l0[i] != min(l0)]
print("{}>{}>{}".format(m1[0], m3[0], m2[0]))
```

## 99乘法表

```python
for row in range(1,10):
    for col in range(1,row+1):
        print("{}*{}={}".format(col,row,col* row),end=" ")
    print()
```

## 字符串替换

```python
x = input()
s = x.replace("ol","fzu")
print(s[::-1])
```

## list

```python
x = [1,"abc",2]
for i in x:
    if type(i) == str:
        x.remove(i)
x.sort()
print(x)
```

## dict

```python
