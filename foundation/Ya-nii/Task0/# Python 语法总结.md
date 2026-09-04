# Python 语法总结

## 1.输入一个字符串

```python
s=input()
print(s)
```

## 2.输入一个整数

```python
n=int(input())
print(n)
```

## 3.输入多个字符串

```python
a,b=input().split()
```

## 4.输入多个整数

```python
a,b=map(int,input().split())
#两个整数 空格分隔
```

```python
nums=list(map(int,input().split()))
#一行多个整数 空格分隔
```

## 5. for 循环

```python
for 变量 in 可迭代对象:
    循环执行的代码
```
