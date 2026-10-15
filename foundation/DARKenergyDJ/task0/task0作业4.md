# task0 作业4

## P1001

```pyhon
a = input().split("")
print(int(s[0])+int(s[1]))
```

## P1046

```python
x = input().split(" ")
y = int(input())
s = 0
for i in x:
    m = int(i)
    if y+30 >= m:
        s += 1
    else:
        continue
print(s)
```

## P5737

```python
x = input().split()
a = int(x[0])
b = int(x[1])
m = 0
s = []
for i in range(a,b+1):
    if i % 400 == 0:
        m += 1
        s.append(i)
    elif i % 100 == 0:
        continue
    elif i % 4 ==0 and i % 400 != 0:
        m += 1
        s.append(i)
    else:
        continue
print(m)
for j in s:
    print(j,end=" ")
```

## ARC017A

```python
N = int(input())
for i in range(1,int(N**0.5)):
    if N % i == 0:
        a = 1
    else:
        a = 0
if a == 1:
    print("YES")
else:
    print("NO")
```

## ACM 学长

```python
n = int(input())
x = []
y = []
v = 0
sum = 0
z= []
for i in range(n):
    s1 = input().split("\n")
    x.extend(s1)
m = int(input())
for j in range(m):
    s2 = input().split("\n")
    y.extend(s2)
for row in y:
    for col in row:
        if col == "1":
            sum += 1
            v = int(row[2])
        else:
            continue
vname = x[v-1]
print("I_love_"*sum,vname)
```