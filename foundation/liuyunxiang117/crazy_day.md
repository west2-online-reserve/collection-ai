# 这是学习 Markdown 的第一天

## Markdown 语法

### 粗体和斜体的用法
**这个是粗体**

*这个是斜体*

***这个是两个一起***

### 引用块
> 嵌套块引用
>
>>大概就是这样

> 带有其它元素的块引用
>
> - 看起来也不错
>
> **和粗体结合一下**

### 列表
有序列表
1. First item
2. Second item
3. Third item
4. Fourth item

无序列表
- First item
- Second item
- Third item
- Fourth item

嵌套引用块
1. First item
2. Second item
   > this
3. Third item

### 代码语法
要用反引号表示，像是`code`这样子
这是九九乘法表
```python
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f'{i}×{j}={i * j}', end='\t')
    print()
```
### 分割线
第一行

——————————————————————————

第二行
### 链接语法
这是一个链接 [blibli](https://www.bilibili.com/)
还可以这样<https://www.bilibili.com/>
### 照片
![风景照](https://ts4.mm.bing.net/th?id=OIP.SxcBYLK1K7nrQa9Js6cvfwHaJQ&pid=15.1)
### 表格

| 姓名 | 性别 | 身高 |
| --- | --- | --- |
| 小明 | 男 | 183 |
| 小红 | 女 | 163 |
| 小张 | 男 | 175 |