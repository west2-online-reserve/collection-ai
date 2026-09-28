x,y,z=map(int,input().split())
if x<y:
    x,y=y,x
if x<z:
    x,z=z,x
if y<z:
    y,z=z,y
print(x,y,z)

#能力有限 其他的只想到穷举法 但是太无聊了就不写了吧


#九九乘法表
a=[]
for i in range(1,10):
    b=[]
    for j in range(1,i+1):
        b.append(f'{i}*{j}={i*j}')
    a.append(b)
print(a)

#ol
a=str(input())
if 'ol' in a:
    a=a.replace('ol','fzu')
a=a[::-1]
print(a)

#列表
a=input().split()
b=[]
for i in a:
    try:
        b.append(int(i))
    except ValueError:
        pass
b.sort()
print(b)


#字典
a={4574:'小明',4587:'小红',4598:'小兰',4579:'小白'}
for i in list(a.keys()):
    if int(i)%2==0:
        del a[i]
print(a)

#函数
def record(l):
    c=0
    d={}
    l=''.join(str(i) for i in l)
    for i in range(1,10):
        c=0
        for j in l:
            if i==int(j): 
                c+=1
        d[i]=c
    print(d)


#商品
class obj:
    def __init__(self,rank,name,money,total,rest):
        self.__rank=rank
        self.__name=name
        self.__money=money
        self.__total=total
        self.__rest=rest

    def display(self):
        d = {'序号': self.__rank,
            '名字': self.__name,
            '单价': self.__money,
            '总量': self.__total,
            '剩余数量': self.__rest}
        print(d)
      
    def income(self):
        v=self.__money*(self.__total-self.__rest)
        print('赚的钱：',v)
        
    def setdata(self,rank,name,money,total,rest):
        self.__rank=rank
        self.__name=name
        self.__money=money
        self.__total=total
        self.__rest=rest


#装饰器
import time, functools

def log(fn):
    @functools.wraps(fn)
    def logger(*s,**g):
        print('begin',f'{fn.__name__}')
        start=time.time()
        r=fn(*s,**g)
        endt=time.time()
        print(r,
              '开始时间：',start,
              '结束时间：',endt,
              '执行时间：',endt-start)
        return r
    return logger

#斗地主
import random
dic={'spade':0.1,'heart':0.2,'club':0.3,'diamond':0.4}
color=['spade','heart','club','diamond']
num=[x for x in range(1,14)]
total=[(i,x) for i in color for x in num]
total+=[('Joker1',17),('Joker2',16)]
random.shuffle(total)
a,b,c=[total[i*17:(i+1)*17] for i in range(3)]
d=total[51:54]
def way(t):
    f=t[1]
    try: 
        s=dic[t[0]]
    except KeyError:
        s=0
    if f==1:
        f=14
    elif f==2:
        f=15
    return f+s
def paixu(t):
    t=sorted(t,key=way,reverse=True)
    return t
with open('player1.txt','w') as f:
    f.write(str(paixu(a))+'\n')
with open('player2.txt','w') as f:
    f.write(str(paixu(b))+'\n')
with open('player3.txt','w') as f:
    f.write(str(paixu(c))+'\n')
with open('others.txt','w') as f:
    f.write(str(paixu(d))+'\n')
#题目要求放在这几个文件里 我只是做题 所以就没建对应的文件


#魔术方法
class MyZoo:
    def __init__(self,animals=None):
        self.animals=animals or {}
        print('My Zoo!')
    def __str__(self):
        return(self.animals)
    def __eq__(self,other):
        return set(self.animals.keys())==set((other.animals.keys()))
    def __len__(self):
        return sum(self.animals.values())

#正则
import re
passwords=str(input())
print(re.match(r'[0-9a-zA-Z]*{6,19}',passwords))





        



        
            

        
    





