#list可变序列，有序，元素可重复
#str不可变序列，有序，字符可重复
#dict可变映射无序
#1.
nums=list(map(int,input().split()))
print(sorted(nums))

num=nums
num.sort()
print(num)

#2.
for i in range (1,10):
    for j in range(1,i+1):
        print(f"{i}*{j}={i*j} ",end="")
    print()
    

#3.
str1=str(input())
a=str1.find("ol")
if (a!=-1):
    str1=str1.replace("ol","fzu")
print(str1[::-1])

#4.
lst=input().split()
new=list()
for element in lst:
    if(element.isdigit()):
        num=int(element)
        new.append(num)
print(sorted(new))

lst=input().split()
for element in lst[:]:
    if not element.isdigit():
        lst.remove(element)
print(sorted(lst))

#5.
d=dict()
while(1):
    num = int(input())
    if num==0:
        break
    name = input()
    d[num] = name

for n in list(d.keys()):
    if n % 2 == 0:
        d.pop(n)

print(d)

#6.
def count(lst,str1):
    cnt=0
    for num in lst:
        if num==str1:
            cnt+=1
    return cnt

string=input().split()
d=dict()
for element in string:
    n=count(string,element)
    k=element
    d[k]=n
print(d)

#7.类预继承
#eg:
# class animal:
#     def __init__(self,name,age):
#         animal.name=name
#         animal.age=age
# class dog(animal):
#     def bark(self):
#         print(f"{self.name} say woof")

# mydog=dog("buddy",3)
# mydog.bark()

class shop:
    def __init__(self,number,name,price,total,rest):
        shop.number=number
        shop.name=name
        shop.price=price
        shop.rest=rest
        shop.total=total
    def display(self):
        print(f"商品序号：{self.number}")
        print(f"商品名称：{self.name}")
        print(f"商品单价：{self.price}")
        print(f"总数量：{self.total}")
        print(f"剩余数量：{self.rest}")
    def income(self):
        print(self.price*(self.total-self.rest))
    def setdata(self,num,na,p,t,r):
        shop.name=na
        shop.rest=r
        shop.price=p
        shop.total=t
        shop.number=num

s1=shop(1,"梳子",5,10,3)
s1.display()
s1.income()
s1.setdata(2,"椅子",30,5,2)
s1.display()

#8.发牌程序(切牌程序，文件读入写入)
#文件读写：
# with open("test.yxt",mode="r",encoding="utf-8")as f:
#     content=f.read()
#     print("tex")
#     print(content)
# writelines表示多行（接受列表和元组）
#write表示一行

import random
card=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,
          5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,
          9,9,9,9,10,10,10,10,"j","j","j",
          "j","q","q","q","q","k","k","k","k",
          "大王","小王"]
random.shuffle(card)
p1=card[:17]
p2=card[17:34]
p3=card[34:51]
others=card[51:]
def save_to_file(filename, cardlist):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, cardlist)))

save_to_file("player1.txt", p1)
save_to_file("player2.txt", p2)
save_to_file("player3.txt", p3)
save_to_file("others.txt", others)

print("玩家1：", p1)
print("玩家2：", p2)
print("玩家3：", p3)
print("底牌：", others)



        

