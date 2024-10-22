import random
num=0
SKILL=[]
skille=[]
skillm=[]
t=0
a,b,c=input("请在1234中输入3个,选择你的宝可梦")
a=int(a)
b=int(b)
c=int(c)
recover=0
step=False
step2=False
detm=False
dete=False
run1=False
gt=0
at=0
time=0
restrain=2
m=0
awa2=[]
e=0#以上为数值（开关）初始化
def choice(a):#角色选择播报
    if a==1:
        print("你选择了小火龙")
    if a==2:
        print("你选择了杰尼龟")
    if a==3:
        print("你选择了妙蛙种子")
    if a==4:
        print("你选择了皮卡丘")
choice(a)
choice(b)
choice(c)
awa1=[1,2,3,4]
def cycle():#单局选择
    global awa,qwq,awa2
    awa=int(input("请选择你本局使用的宝可梦，不要选择已经使用过的宝可梦"))
    while awa in awa2:
        awa=int(input("你的宝可梦已经使用过了，请重新选择"))
    awa2.append(awa)
    
    

    while 1:
        qwq=random.choice(awa1)
        if qwq==awa:
            qwq=random.choice(awa1)
        else:
            break
    awa1.remove(qwq)
    return awa,qwq
cycle()
choice(awa)
def choicee():
    global qwq
    if qwq==1:
        print("电脑选择了小火龙")
    if qwq==2:
        print("电脑选择了杰尼龟")
    if qwq==3:
        print("电脑选择了妙蛙种子")
    if qwq==4:
        print("电脑选择了皮卡丘")
choicee()
def ass():
    global Charmander,pikaqiu,Squirtle,Bulbasaur
    Charmander=[80,'fire',35,15,10]#0hp 1属性 2伤害 3防御 4闪避
    pikaqiu=[80,'flash',35,5,30]#皮卡车
    Squirtle=[80,'water',25,20,20]#杰尼龟
    Bulbasaur=[100,'grass',35,10,10]#妙蛙种子
ass()
C=['Ember','Flame Charge']
P=['Thunderbolt','Quick Attack']
S=['Aqua Jet','Shield']
B=['Seed Bomb','Parasitic Seeds']

class PROCESS():
    def skill(self,name,position):
        print(f"{position}使用了{name}")
    def run(self,awa,position):
        global hurte,run1
        NUM=[]
        for i in range(awa[4]):
            NUM.append(i+1)
        x=0
        for i in range(100):
            rate=random.randint(1,100)
            if rate in NUM:
                x+=1 
        if x==awa[4]:
            print(f"{position}闪避了该技能")#技能SKILL[skill1-1]
            hurte=0
            run1=True
            
        else:    
            print(f"{position}没有闪避该技能")
        return hurte
    def feature(self,skillm,skill1,awa):
        global hurte,recover,step,step2
        
        if skillm[skill1-1]=="Ember":
            hurte=awa[2]
            NUM=list(range(10))
            x=0
            if awa[4]==30:
                awa[4]-=20
            for _ in range(100):

                rate=random.randint(1,100)
                if rate in NUM:
                    x+=1 
                if x==10:
                    print("敌方被烧伤")
                break
            else:
                pass
        if skillm[skill1-1]=="Flame Charge":
            hurte=awa[2]
            awa[4]+=20
        #还需要增加地方20闪避(完成)
        if skillm[skill1-1]=="Thunderbolt":
            hurte=1.4*awa[2]
            NUM=list(range(10))
            x=0
            for _ in range(100):

                rate=random.randint(1,100)
                if rate in NUM:
                    x+=1 
                if x==10:
                    print("敌方被麻痹")#麻痹的特性
                    step=True
                    break
        if skillm[skill1-1]=="Quick Attack":
            hurte=awa[2]
            NUM=list(range(10))
            x=0
            for _ in range(100):

                rate=random.randint(1,100)
                if rate in NUM:
                    x+=1 
                if x==10:
                    print("触发第二次攻击")
                    
                    hurte=2*awa[2]
                    break
        if skillm[skill1-1]=="Seed Bomb":
            hurte=awa[2]
            NUM=list(range(15))
            x=0
            for _ in range(100):

                rate=random.randint(1,100)
                if rate in NUM:
                    x+=1 
                if x==15:
                    print("敌方中毒了")#中毒特性
                    step2=True
                    break
        if skillm[skill1-1]=="Parasitic Seeds":
            hurte=awa[2]
            recover=awa[2]#造成同等伤害的吸血
        if skillm[skill1-1]=='Aqua Jet':
            hurte=1.4*awa[2]
        if skillm[skill1-1]=='Shield':
            hurte=0
            step=True#修改为必定躲避一次攻击
       
        return hurte
    def fight(self,hp,position,defence,recover,hurte):
        global at,gt,restrain,step
        if time==0:
            if at=="f":#修改为火元素有百分之三十的概率暴击 造成1，5倍伤害
                if random.random()<=0.3:
                    hurte*=1.5
            if hurte>0:
                if at=="w":
                    if random.random()<=0.5:
                        hurte*=0.7  
            if at=="g":
                hp*=1.1
            if gt=='fl':
                if random.random()<=0.3:
                    step=True
                    print("敌方飞起来了,逃掉一次攻击")
            
            if restrain==1:
                hurte*=1.3#翻倍伤害过高 调整为压制时伤害为1.3倍 
            if restrain==0:
                hurte*=0.7




        if time==1:
            if gt=="f":#修改为火元素有百分之三十的概率暴击 造成1，5倍伤害
                if random.random()<=0.3:
                    hurte*=1.5
            if hurte>0:
                if gt=="w":
                    if random.random()<=0.5:
                        hurte*=0.7  
            if gt=="g":
                hp*=1.1
            if at=='fl':
                if random.random()<=0.3:
                    step=True
                    print("你飞起来了,逃掉一次攻击")
            if restrain==1:
                hurte*=0.7#翻倍伤害过高 调整为压制时伤害为1.3倍 
            if restrain==0:
                hurte*=1.3

                
            #加入顺序系统 轮流判断敌方和我方属性 并加入while循环 
        
        
                





        hurte-=defence
        if hurte<=0:
            hurte=0
        hp-=hurte
        hp+=recover
        if step2==True:
            hp*=0.9
        hp=int(hp)
        
        print(f"{position}的血量还剩下{hp}")
        at=0
        gt=0
        return hp
    def determine(self,det):
        if step==True:
            det=True
        return det
    def attribute(self):
        global at,restrain,gt
        if awa[1]=="fire":
            at="f"
            if qwq[1]=="grass":
                print('你的属性克制对方，伤害提高为130%')
                gt='g'
                restrain=1
            if qwq[1]=='water':
                print("你被敌方克制，伤害降低为%70")
                gt='w'
                restrain=0
            else: 
                gt='fl'
        if awa[1]=="grass":
            at="g"
            if qwq[1]=="water":
                print('你的属性克制对方，伤害提高为130%')
                gt='w'
                restrain=1
            if qwq[1]=='fire':
                print("你被敌方克制，伤害降低为%70")
                gt='f'
                restrain=0
            else:
                gt="fl"
        if awa[1]=="water":
            at="w"
            if qwq[1]=="fire":
                print('你的属性克制对方，伤害提高为130%')
                gt='f'
                restrain=1
            if qwq[1]=='flash':
                print("你被敌方克制，伤害降低为%70")
                gt='fl'
                restrain=0
            else:
                gt='g'
        if awa[1]=="flash":
            at="fl"
            if qwq[1]=="water":
                print('你的属性克制对方，伤害提高为130%')
                gt='w'
                restrain=1
            if qwq[1]=='grass':
                print("你被敌方克制，伤害降低为%70")
                restrain=0
                gt='g'
            else:
                gt='f'
        return at,gt
        
   
def choice1(awa,SKILL):
    if awa==1:
        awa=Charmander
        SKILL=C
    if awa==2:
        awa=Squirtle
        SKILL=S
    if awa==3:
        awa=Bulbasaur
        SKILL=B
    if awa==4:
        awa=pikaqiu
        SKILL=P
    return awa,SKILL
awa,skillm=choice1(awa,skillm)
qwq,skille=choice1(qwq,skille)

print(skillm)
while t<3: #游戏开始
    time=0
   
    skill1=int(input("使用你的技能（1or2）"))
    skill2=random.randint(1,2)
    process=PROCESS()
    process.skill(skillm[skill1-1],"你")
    process.skill(skille[skill2-1],"电脑")
    
    process.feature(skillm,skill1,awa)
    at,gt=process.attribute()
    # print(at)
    # print(gt)
    
    
    hurte=process.run(qwq,"敌人")
    dete=process.determine(dete)
    step=False
    if detm==True:
        hurte=0
    # print(detm)
    detm=False
    
    
    print(f'造成的伤害为{hurte}')
    qwq[0]=process.fight(qwq[0],"电脑",qwq[3],recover,hurte)
   
    if qwq[0]<=0:
        step2=False
        print("敌方宝可梦昏厥了")

        m+=1
        t+=1
        if t==3:
            break
        ass()
        cycle()
        choice(awa)
        choicee()
        awa,skillm=choice1(awa,skillm)
        qwq,skille=choice1(qwq,skille)
        
        continue



    time=1
    run1=False
    process.feature(skille,skill2,qwq)
    detm=process.determine(detm)
    # print(detm)
    
    hurte=process.run(awa,"你")
    if dete==True:
        hurte=0
    step=False
    dete=False
    print(f'造成的伤害为{hurte}')
    awa[0]=process.fight(awa[0],"你",awa[3],recover,hurte)
    run1=False
    
    if awa[0]<=0:
        step2=False
        print('你的宝可梦昏厥了')
        e+=1
        t+=1
        if t==3:
            break   
        ass()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        cycle()
        choice(awa)
        choicee()
        awa,skillm=choice1(awa,skillm)
        qwq,skille=choice1(qwq,skille)
        choicee()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        continue

    recover=0                                                                                                                                                                                      
if e>m:
    print("敌人赢得了这次比赛")
else:
    print("你赢的了这次比赛")
   

    






  


                
                

                





    