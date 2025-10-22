#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#引入模块
from time import sleep;import random;from sys import exit
#缩减写法
pr,inp,sl=print,input,sleep

#一些提前写好的变量
play=[];compu=[]
fight_poke='wu';compu_fight_poke='wu'
max_sta_level=10
turn=0
end1=0

#属性克制表
ty_list={
    '草':{'水':'克制','火':'被克制','兽':'被克制','神':'被克制'},
    '火':{'草':'克制','水':'被克制','兽':'被克制','神':'被克制'},
    '水':{'火':'克制','电':'被克制','兽':'被克制','神':'被克制'},
    '电':{'水':'克制','草':'被克制','兽':'克制','神':'被克制'},
    '神':{'草':'克制','水':'克制','火':'克制','电':'克制','兽':'克制'},
    '兽':{'草':'克制','水':'克制','火':'克制','电':'被克制','神':'被克制'}
}
#状态列表(明/暗)
sta_ty=['麻痹','中毒','防护','烧伤','寄生','神化','天诛'];de_sta_ty=['防护精华','麻痹精华','神化精华']

#基类
class Pokenmon(object):
    '''宝可梦基类'''
    def __init__(self,name,hp,at,de,mi_ra,ty,num):
        self.name=name#名字
        self.hp=hp#当前生命值
        self.at=at#攻击力
        self.de=de#防御力
        self.mi_ra=mi_ra#闪避率
        self.ty=ty#属性
        self.max_hp=hp#最大生命值
        self.num=num#编号
        self.mis=True#闪避开启开关
        self.sta = {state: 0 for state in sta_ty + de_sta_ty}#属性列表

    #对敌人状态函数
    def burning(self,enemy,rate,level,aton):
        if aton:
            if random.randint(1,100)<=rate:
                pr(f'{enemy.name} 被烧伤了!')
                return '烧伤',level
        return '',level
        
    def posion(self,enemy,rate,level,aton):
        if aton:
            if random.randint(1,100)<=rate:
                pr(f'{enemy.name} 中毒了!')
                return '中毒',level
        return '',level
    
    def parasi(self,enemy,rate,level,aton):
        if aton:
            if random.randint(1,100)<=rate:
                pr(f'{enemy.name} 被寄生了!')
                return '寄生',level   
        return '',level

    def paraly(self,enemy,rate,level,aton):
        if aton:
            if random.randint(1,100)<=rate:
                pr(f'{enemy.name} 被麻痹了!')
                return '麻痹精华',level
        return '',level

    #对自身状态函数
    def barricade(self,rate,level):
        if rate>=random.randint(1,100):

            return '防护精华',level
        else:
            return '',level
    
    def deification(self,rate,level):
        if rate>=random.randint(1,100):
            return '神化精华',level
        else:
            return '',level
    
    def god_anger(self,rate,level):
        if rate>=random.randint(1,100):
            pr(f'{self.name}获得了一层“天诛”')
            return '天诛',level
        else:
            pr(f'{self.name}')
            return '',level

    #其他函数    
    def double_att(self,rate): 
        if random.randint(1,100)<=rate:
            pr(self.name,'触发了连击!')
            return True
        return False

#创建属性工厂函数
def creat_ty(cls_name,ty_name):
    '''
    创建新属性的函数:

    Args:
        cls_name:属性类的类名
        ty_name:属性名称
    
    Returns:
        新的属性类
    '''
    class Newtype(Pokenmon):
        def __init__(self,name,hp,at,de,mi_ra,num):
            super().__init__(name=name,hp=hp,at=at,de=de,mi_ra=mi_ra,ty=ty_name,num=num)
    Newtype.__name__=cls_name
    return Newtype
            

#不同属性    
Dian=creat_ty('Dian','电')
Cao=creat_ty('Cao', '草')
Shui=creat_ty('Shui', '水')
Huo=creat_ty('Huo', '火')
Beast=creat_ty('Beast', '兽')
God=creat_ty('God', '神')

#具体宝可梦

#1:皮卡丘
class pikachu(Dian):
    def __init__(self):
        super().__init__(name='皮卡丘',hp=80,at=35,de=5,mi_ra=30,num=1)#30
        self.skills={'1:十万伏特(释放超强电流!对敌方造成 1.4 倍攻击力的电属性伤害，并有 15% 概率使敌人麻痹)':self.Thunderbolt,
                     '2:电光一闪(释放一道迅捷的闪电,对敌方造成 1.0 倍攻击力的快速攻击， 10% 概率再触发一次闪光一击（至多五次）':self.Quick_Attack}

    def Thunderbolt(self,enemy,**kw):#
        damage=int(self.at*1.4)
        sta1,level=self.paraly(enemy,15,1,kw['aton'])
        pr(f'\n{self.name} 使用了 十万伏特!')
        return damage,sta1,level,'',0
    
    def Quick_Attack(self,enemy,hit_count=1,**kw): 
        damage=self.at
        if hit_count<5 and self.double_att(10):
            extra_damage,_,_,_,_=self.Quick_Attack(enemy,hit_count+1,**kw)
            damage+=extra_damage
        fixed=f'\n{self.name} 使用了 电光一闪!'

        if hit_count==1:
            pr(fixed+'(一破,卧龙出山!)')
        elif hit_count==2:
            pr(fixed+'(双连,一战成名!)')
        elif hit_count==3:
            pr(fixed+'(三连,举世皆惊!)')
        elif hit_count==4:
            pr(fixed+'(四连,天下无敌!)')
        else:
            pr(fixed+'(五连,诸天灭地!)')


        return damage,'',0,'',0
#2:妙蛙种子    
class bulbasaur(Cao):
    def __init__(self):
        super().__init__(name='妙蛙种子',hp=100,at=35,de=10,mi_ra=10,num=2)#10
        self.skills={'1:种子炸弹(发射一颗种子，爆炸后对敌方造成草属性伤害。若击中目标，目标有15%几率陷入“中毒”状态，每回合损失10%生命值)':self.Seed_Bomb,
                     '2:寄生种子(向敌方播种，每回合吸取敌方10%的最大生命值并恢复自己， 效果持续3回合)':self.Parasitic_Seeds}

    def Seed_Bomb(self,enemy,**kw):
        damage=self.at
        pr(f'\n{self.name} 使用了 种子炸弹!')
        sta1,level=self.posion(enemy,15,1,kw['aton'])
        return damage,sta1,level,'',0
    
    def Parasitic_Seeds(self,enemy,**kw):
        damage=0
        sta1,level1=self.parasi(enemy,100,3,aton=True)
        pr(f'\n{self.name} 使用了 寄生种子!')
        return damage,sta1,level1,'',0
  #      
#3:杰尼龟
class squirtle(Shui):
    def __init__(self):
        super().__init__(name='杰尼龟',hp=80,at=25,de=20,mi_ra=20,num=3)#20
        self.skills={'1:水枪(喷射出一股强力的水流，对敌方造成 140% 水属性伤害)':self.Aqua_Jet,
                     '2:护盾(使用水流形成保护盾，下一回合受到的伤害-50%)':self.Shield}

    def Aqua_Jet(self,enemy,**kw):
        damage=int(self.at*1.4)
        pr(f'\n{self.name} 使用了 水枪!')
        return damage,'',0,'',0
    
    def Shield(self,enemy,**kw):#%
        damage=0
        sta2,level2=self.barricade(100,1)
        pr(f'\n{self.name} 使用了 护盾!下回合受到的伤害减少 50%')
        return damage,'',0,sta2,level2
#4:小火龙
class charmander(Huo):
    def __init__(self):
        super().__init__(name='小火龙',hp=80,at=35,de=15,mi_ra=10,num=4)#10
        self.skills={'''1:火花(发射出一团小火焰，对敌方造成 100% 火属性伤害，
                        并有10%的几率使目标获得两层“烧伤”状态（本回合受到10点不计防御的额外伤害）)''':self.Ember,
                     '''2:蓄能爆炎(小火龙召唤出强大的火焰，对敌方造成 300% 火属性伤害，有80%的几率使敌人获得一层“烧伤”状态，
                        这个技能需要1个回合的蓄力，但下个回合敌人的闪避率增加 20%)''':self.Flame_Charge}
        self.fla_time=0


    def Ember(self,enemy,**kw):
        damage=self.at
        sta1,level1='',0
        if self.fla_time>0:
            self.fla_time-=1
            enemy.mi_ra=max(enemy.mi_ra-20*self.fla_time,0)
        pr(f'\n{self.name} 使用了 <火花>!')
        sta1,level1=self.burning(enemy,10,2,kw['aton'])       
        return damage,sta1,level1,'',0 
    
    def Flame_Charge(self,enemy,**kw):
        damage=0
        sta1,level1='',0
        if self.fla_time==1:
            enemy.mi_ra=max(enemy.mi_ra-20*self.fla_time,0)
            damage=self.at*3
            pr(f'\n{self.name} 使用了 <蓄能爆炎>!')
            sta1,level1=self.burning(enemy,80,2,kw['aton'])
            self.fla_time=0
        else:
            self.fla_time+=1
            enemy.mi_ra+=20
            level1=0
            pr(f'{self.name} 的 <蓄能爆炎> 正在蓄力')
        return damage,sta1,level1,'',0
#5:林鸦
class fores_crow(Cao):
    def __init__(self):
        super().__init__(name='林鸦',hp=80,at=25,de=5,mi_ra=30,num=5)#30
        self.skills={'1:指指(向低指了一指,对敌方造成了100%草属性伤害,闪避率永久增加5,至多三次)':self.Pointin,
                     '2:点点(向天点了一点,对敌方造成等同于自身HP的草属性伤害\n同时有 90% 概率获得一层“天诛”状态(下回合结束时死亡))':self.Countin}
        self.point_times=0


    def Pointin(self,enemy,**kw):
        damage=self.at
        pr(f'{self.name} 使用了 <指指>!')
        if self.point_times<3:
            self.point_times+=1
            self.mi_ra+=5
            pr(f'{self.name}的闪避率增加了5%!')
        else:
            pr('闪避率增加次数到达上限!')   
        return damage,'',0,'',0 
    
    def Countin(self,enemy,**kw):
        damage=self.hp
        sta2,level2=self.god_anger(90,1)
        enemy.mis=False
        pr(f'{self.name} 使用了 <点点>!上天震怒,降下了雷暴!双方均无法闪避!')
        return damage,'',0,sta2,level2
#6:神-槐安
class god_huai(God):
    def __init__(self):
        super().__init__(name='神-槐安',hp=80,at=20,de=15,mi_ra=25,num=6)#25
        self.skills={'1:飞升(修炼自己,提高了境界!\n获得一层“神化”状态(下一次受到的伤害减 10))':self.Ascendin,
                     '2:神堕(释放愤怒,道心不稳!对敌方造成攻击力+10×使用过飞升次数点 无视闪避的伤害,然后视为使用过的飞升次数清零)':self.Corruptin
        }
        self.ascend_times=0

    def Ascendin(self,enemy,**kw):
        damage=0
        sta2,level2=self.deification(100,1)  
        self.ascend_times+=1
        pr(f'{self.name} 使用了 <神化>!下次受到的伤害减 10')  
        return damage,'',0,sta2,level2 
    
    def Corruptin(self,enemy,**kw):
        damage=self.at+10*self.ascend_times
        self.ascend_times=0
        enemy.mis=False
        pr(f'{self.name} 使用了 <神堕>!{enemy.name} 无法闪避!')
        return damage,'',0,'',0
#7:塞布兔
class sevniao_rabbit(Beast):
    def __init__(self):
        super().__init__(name='塞布兔', hp=70, at=25, de=20, mi_ra=35, num=7)
        
        # 新概念：月光能量系统
        self.moon_energy = 0  # 月光能量 (0-100)
        self.charm_points = 0  # 魅力值 (影响技能效果)
        
        self.skills = {
            '1:月亮之力(凝聚月光能量，造成 120% 伤害并恢复10%月光能量(每回合自动恢复5%,最高100%)。根据月光能量额外提升伤害)':self.Lunar_power,
            '2:魅惑之声(用迷人的声音迷惑对手，造成 80% 伤害并提升自身魅力值。魅力值越高，麻痹概率越高)':self.Charmin_voice,
            '3:吸取之吻(发射魔法亲吻，造成 90% 伤害并吸取敌人15%造成伤害的生命值，同时转移5点魅力值)':self.Absorbin_kis,
            '4:月夜狂欢(消耗所有月光能量，每10点能量造成额外5%伤害，并重置魅力值获得且新效果:\n 高魅力(≥30):获得防护状态 | 中魅力(15-29):恢复20%生命 | 低魅力(<15):仅造成伤害)':self.Moonlight_frenzy
        }

    def Lunar_power(self, enemy, **kw):
        """月亮之力 - 基于月光能量的强化攻击"""
        base_damage = int(self.at * 1.2)
        
        # 月光能量加成：每10点能量增加2%伤害
        energy_bonus = int(base_damage * (self.moon_energy // 10) * 0.02)
        total_damage = base_damage + energy_bonus
        
        # 恢复月光能量
        energy_gain = 10
        self.moon_energy = min(self.moon_energy + energy_gain, 100)
        
        pr(f'\n{self.name} 使用了 月亮之力!')
        pr(f'月光能量: {self.moon_energy}/100 (+{energy_gain})')
        if energy_bonus > 0:
            pr(f'月光加成: +{energy_bonus} 伤害!')
        
        return total_damage, '', 0, '', 0
    
    def Charmin_voice(self, enemy, **kw):
        """魅惑之声 - 提升魅力并概率麻痹"""
        damage = int(self.at * 0.8)
        
        # 提升魅力值
        charm_gain = 15
        self.charm_points = min(self.charm_points + charm_gain, 50)
        
        # 魅力值影响麻痹概率：基础10% + 每5点魅力增加5%
        paraly_rate = 10 + (self.charm_points // 5) * 5
        sta1, level = self.paraly(enemy, paraly_rate, 1, kw['aton'])
        
        pr(f'\n{self.name} 使用了 <魅惑之声>!')
        pr(f'魅力值: {self.charm_points}/50 (+{charm_gain})')
        if self.charm_points >= 20:
            pr(f'✨ 魅力四射! 麻痹概率提升至 {paraly_rate}%')
        
        return damage, sta1, level, '', 0
    
    def Absorbin_kis(self, enemy, **kw):
        """吸取之吻 - 吸血并转移魅力"""
        damage = int(self.at * 0.9)
        
        # 吸血效果：造成伤害的15%
        drain_amount = int(damage * 0.15)
        
        # 魅力转移：从敌人那里偷取魅力（如果有的话）
        charm_steal = 5
        if hasattr(enemy, 'charm_points') and enemy.charm_points > 0:
            actual_steal = min(charm_steal, enemy.charm_points)
            enemy.charm_points -= actual_steal
            self.charm_points = min(self.charm_points + actual_steal, 50)
            pr(f'💋 偷取了 {actual_steal} 点魅力值!')
        else:
            self.charm_points = min(self.charm_points + charm_steal, 50)
        
        pr(f'\n{self.name} 使用了 <吸取之吻>!')
        pr(f'吸取了 {drain_amount} 点生命值!')
        
        # 返回吸血量给调用者处理
        return damage,'', 0,'',drain_amount
    
    def Moonlight_frenzy(self, enemy, **kw):
        """月夜狂欢 - 消耗所有能量的爆发技能"""
        if self.moon_energy < 20:
            pr(f'\n{self.name} 的月光能量不足! 需要至少20%能量。')
            return 0, '', 0, '', 0
        
        #基础数值
        base_damage = self.at
        sta2,level2='',0
        
        # 能量加成：每10点能量增加5%伤害
        energy_multiplier = 1 + (self.moon_energy // 10) * 0.05
        total_damage = int(base_damage * energy_multiplier)
        
        # 魅力值重置效果
        old_charm = self.charm_points
        if old_charm >= 30:
            # 高魅力：获得防护状态
            self.charm_points = 0
            pr(f'🌟 魅力爆发! 获得防护状态!')
            sta2,level2='防护',1
        elif old_charm >= 15:
            # 中魅力：恢复生命值
            heal_amount = int(self.max_hp * 0.2)
            self.charm_points = 0
            pr(f'✨ 魅力转化! 恢复 {heal_amount} 点生命值!')
            level2=heal_amount
        else:
            # 低魅力：简单重置
            self.charm_points = 0
            pr(f'🌙 月光爆发! 消耗所有能量!')
        
        # 消耗所有月光能量
        energy_used = self.moon_energy
        self.moon_energy = 0
        
        pr(f'\n{self.name} 使用了 月夜狂欢!')
        pr(f'消耗 {energy_used}% 月光能量，造成毁灭性打击!')
        
        return total_damage,'',0, sta2,level2
    
    def end_turn_effect(self):
        """回合结束时的被动效果（需要在主循环中调用）"""
        pr(f'{self.name} 当前月光能量为:{self.moon_energy}%')
        # 每回合自动恢复5点月光能量
        if self.moon_energy < 100:
            self.moon_energy = min(self.moon_energy + 5, 100)
            pr(f'🌙 {self.name} 吸收了月光能量 (+5%)')


#宝可梦列表
poke_list=[pikachu, bulbasaur, squirtle, charmander,fores_crow,god_huai,sevniao_rabbit]


#查看宝可梦函数
def check_poke(poke_l=poke_list):
    while True:
        
        pr('\n宝可梦图鉴:')
        for poke in poke_l:
            poke0=poke()
            pr(f'{poke0.num}:{poke0.name}({poke0.ty}属性)')
        back_opt=str(len(poke_l)+1)
        opt_check=inp(f'{back_opt}:返回\n请输入你要查看的宝可梦的编号(或输入{back_opt}返回):')
        if opt_check in [str(x) for x in range(1,len(poke_l)+1)]:
            c=poke_l[int(opt_check)-1]()
            pr(f'\n★{c.name}★\n==========属性:\n{c.ty}\nHP:{c.hp}\n攻击力:{c.at}\n防御力:{c.de}\n闪避率:{c.mi_ra}\n==========技能:')
            for i in c.skills.keys():
                pr(i)
            sl(1)
        elif opt_check==back_opt:
            break
        else:
            pr('\n请输入有效的数字')
            sl(0.5)
        #text('查看宝可梦循环')

#选宝可梦函数
def cho_poke(poke_list=poke_list):
    play=[]
    compu=[]    
    compu_cho=random.sample(range(1,len(poke_list)+1),3)
    compu_cho=[str(i) for i in compu_cho]

    pr('\n请输入3个不同数字,选择3个不同的宝可梦\n')
    while True:
        
        pr('='*20+'\n')
        for poke in poke_list:
            poke0=poke()
            pr(f'编号- {poke0.num}:{poke0.name}({poke0.ty}属性)')
        play_cho=inp('(每个数字之间需留出一个空格):\n').split()

        if len(play_cho)!=3:
            pr('请输入正好3个数字!')
            sl(0.5)
            continue
        if len(set(play_cho))!=3:
            pr('请输入3个不同数字!')
            sl(0.5)
            continue
        valid=True
        for i in play_cho:
            if not i.isdigit() or i not in [str(x) for x in range(1,len(poke_list)+1)]:
                valid=False
                break
        if valid:
            break
        pr('请输入3个不同数字,选择3个不同的宝可梦!!!\n'+'='*20)
        #text('选宝可梦循环')
        
    
    for i in play_cho:
        for poke in poke_list:
            poke0=poke()
            if i==str(poke0.num):
                play.append(poke0)

    for i in compu_cho:
        for poke in poke_list:
            poke0=poke()
            if i==str(poke0.num):
                compu.append(poke0)
    return play,compu

#战斗函数
def fight(me,enemy,skill_name):
    #战斗开始的一些计算
    mis=random.randint(1,100)
    aton0=mis>enemy.mi_ra
    rise_hp=0
    #计算基础伤害
    ba_damage0,enemy_sta,enemy_sta_level,me_sta,me_sta_level=me.skills[skill_name](enemy,aton=aton0)
    if isinstance(me,sevniao_rabbit) and me.skills[skill_name].__name__=='Absorbin_kis':
        rise_hp=me_sta_level
    sl(0.5)
    ba_damage=max(0,ba_damage0)
    cur_damage=ba_damage-enemy.de
    #结算神化状态
    if enemy.sta['神化']>0:
        pr(f'{enemy.name}正在神化!受到的伤害-10!')
        sl(0.5)
        cur_damage=max(cur_damage-10,0)
    #计算属性克制
    kezhi='普通'
    if enemy.ty in ty_list[me.ty].keys():
        kezhi=ty_list[me.ty][enemy.ty]
    if kezhi=='克制':
        pr(f'{me.name}的{me.ty}属性克制了{enemy.name}的{enemy.ty}属性,造成伤害增加20%!')
        damage=cur_damage+int(cur_damage*0.2)
    elif kezhi=='被克制':
        pr(f'{me.name}的{me.ty}属性被{enemy.name}的{enemy.ty}属性克制,造成伤害减少20%!')
        damage=cur_damage-int(cur_damage*0.2)
    else:
        damage=cur_damage
    #结算麻痹 防护状态
    fin_damage=max(damage,0)
    if me.sta['麻痹']>0:
        pr(f'{me.name} 被麻痹了!无法造成伤害!')
        sl(0.5)
        fin_damage=0
    if enemy.sta['防护']>0:
        pr(f'{enemy.name}进行了防护!受到的伤害-50%!')
        sl(0.5)
        fin_damage=cur_damage//2
    #结算闪避
    if not aton0 and me.sta['麻痹']==0 and enemy.mis:
        fin_damage=0
        pr(f'\n{enemy.name}躲开了!')
        sl(0.5)
    #战斗结束的一些计算
    enemy.mis=True

    return fin_damage,enemy_sta,enemy_sta_level,me_sta,me_sta_level,rise_hp

#烧伤 中毒 寄生 天诛
def hp_del(me,enemy):
    total_decl=0
    enemy_rise=0
    if me.sta['烧伤']>0:
        if me.sta['防护']>0:
            pr(f'{me.name} 进行了防护!受到的伤害-50%')
            damage=5
        else:
            damage=10
        pr(f'{me.name} 被烧伤了,受到{damage}点伤害!')
        total_decl+=damage
    if me.sta['中毒']>0:
        zhongdu_decl=int(me.max_hp*0.1)
        pr(f'{me.name} 中毒了,减少10%({zhongdu_decl}点)的生命值!')
        total_decl+=zhongdu_decl
    if me.sta['寄生']>0:
        jisheng_decl=int(me.max_hp*0.1)
        if enemy.hp==0:
            jisheng_rise=0
        else:
            jisheng_rise=jisheng_decl 
        pr(f'{me.name} 被寄生了,失去了10%({jisheng_decl}点)的生命值!\n{enemy.name} 吸收了{jisheng_rise}点生命值!')
        enemy_rise=min(jisheng_rise,enemy.max_hp-enemy.hp)
        total_decl+=jisheng_decl
    if me.sta['天诛']>0:
        me.hp=0
        pr(f'{me.name} 被天诛!生命值归零')

    if me.hp<0:
        me.hp=0
    return total_decl,enemy_rise
    
#状态层数减少
def sta_dec(poke,list):
    for sta in list:
        if poke.sta[sta]>0:
            poke.sta[sta]-=1
            if poke.sta[sta]==0:
                pr(f'{poke.name}的{sta}状态完全消失了!\n')
            else:
                pr(f'{poke.name}的{sta}减少了一层!\n')
            sl(0.05)

#结束函数
def end_(end1):
    if end1==1:
        pr('\n在敌人的猛烈攻势下,你的宝可梦完全不敌,战败了...')
    elif end1==2:
        pr('\n你和你的宝可梦齐心协力,打倒了敌人!')
    else:
        pr('\n出乎意料,你们几乎势均力敌,最终不分胜负!')
    inp('输入任意键退出\n')

#测试函数
def text(loop_name='未知循环'):
    if not hasattr(text,'ready'):
        text.ready=True
        pr('\n开始测试是否有无意义循环')
        sl(0.1)

    tex=inp('\n发现无意义循环!输入e结束循环:')
    if tex.lower()=='e':
        pr(f'测试结束,{loop_name} 出错!')
        sl(0.5)
        exit()


#欢迎
pr('欢迎来到文字宝可梦!')

#检查宝可梦
check_opt=inp('你需要查看宝可梦图鉴吗?\n若需要,请输入 c (不需要请直接按下回车键):')
if check_opt=='c':
    check_poke()


#正式开始

#选择宝可梦
play,compu=cho_poke()

#核心战斗主循环
while True:
    #选择出战的宝可梦
    if fight_poke=='wu':
        if not play:
            end1=1
            break
        pr('\n请输入数字选择你要出战的宝可梦:')
        sl(0.2)
        opt=[]
        
        for i in range(len(play)):
            sl(0.1)
            opt.append(f'{i+1}:{play[i].name}({play[i].ty}属性)')
        while True:
            fight_poke_num=inp(' '.join(opt)+'\n')
            if fight_poke_num.isdigit() and 1<=int(fight_poke_num)<=len(play):
                break
            pr('\n输入错误,请重新选择有效的数字!')
            sl(0.3)
            #text('选择出战宝可梦循环')
        fight_poke=play[int(fight_poke_num)-1]
        pr(f'\n你选择了 {fight_poke.name}')

    if compu_fight_poke=='wu':
        if not compu:
            end1=2
            break
        compu_fight_poke=compu[random.randint(0,len(compu)-1)]
        pr(f'\n敌人选择了 {compu_fight_poke.name}')

    
    #开始战斗
    turn+=1
    sl(0.5)

    #回合开始
    pr(f'\n--------回合{turn}--------')

    if isinstance(fight_poke, sevniao_rabbit):
        fight_poke.end_turn_effect()
    if isinstance(compu_fight_poke, sevniao_rabbit):
        compu_fight_poke.end_turn_effect()

    #展示宝可梦血量
    pr(f'你的 {fight_poke.name}:{fight_poke.hp}/{fight_poke.max_hp}')
    pr(f'敌方的 {compu_fight_poke.name}:{compu_fight_poke.hp}/{compu_fight_poke.max_hp}')

    #结算暗状态
    for i in [fight_poke,compu_fight_poke]:
        if i.sta['防护精华']>0:
            i.sta['防护']+=1
        if i.sta['麻痹精华']>0:
            i.sta['麻痹']+=1
        if i.sta['神化精华']>0:
            i.sta['神化']+=1 
        sta_dec(i,de_sta_ty)

    #展示明状态层数
    wu=True
    pr(f'\n你的{fight_poke.name}的状态:')
    for i in sta_ty:
        if fight_poke.sta[i]>0:
            wu=False
            sl(0.05)
            pr(f'{i}:{fight_poke.sta[i]}层')
    if wu:
        pr('无')
    
    wu=True
    pr(f"\n敌人的{compu_fight_poke.name}的状态:")
    for i in sta_ty:
        if compu_fight_poke.sta[i]>0:
            wu=False
            sl(0.05)
            pr(f'{i}:{compu_fight_poke.sta[i]}层')
    if wu:
        pr('无')


    #你的技能
    skill1=[]
    sl(0.5)
    pr(f'\n你的  {fight_poke.name}  的技能')
    for x in fight_poke.skills:
        sl(0.1)
        pr(x)
        skill1.append(x)
    
    while True:
        move1=inp('\n输入数字,选择一个技能进行战斗:')
        if move1.isdigit() and move1 in [str(x) for x in range(1,len(fight_poke.skills)+1)]:
            break
        pr(f'请输入 1-{len(fight_poke.skills)} 之间的数字!')
        sl(0.2)
    #你的攻击

    #选择技能
    move1=int(move1)
    selected_skill_name1=skill1[move1-1]
    #计算战斗结果
    fin_damage,compu_sta,compu_sta_level,play_sta,sta_level,rise_hp=fight(fight_poke, compu_fight_poke, selected_skill_name1)
    #血量改变
    compu_fight_poke.hp-=fin_damage
    fight_poke.hp+=rise_hp
    #状态层数最多为10
    if compu_sta in sta_ty+de_sta_ty:
        compu_fight_poke.sta[compu_sta]=min(compu_fight_poke.sta[compu_sta]+sta_level,max_sta_level)
    if play_sta in sta_ty+de_sta_ty:
        fight_poke.sta[play_sta]=min(fight_poke.sta[play_sta]+sta_level,max_sta_level)
    #血量最低为0
    if compu_fight_poke.hp<=0:
        compu_fight_poke.hp=0
    #显示战斗结果
    if fin_damage>0:
        pr(f'\n敌方的 {compu_fight_poke.name} 受到了 {fin_damage} 点伤害! 剩余HP:{compu_fight_poke.hp}/{compu_fight_poke.max_hp}\n')
    else:
        pr(f'\n敌方的 {compu_fight_poke.name} 没有受到伤害!剩余HP:{compu_fight_poke.hp}\n')
    if rise_hp>0:
        pr(f'\n你的 {fight_poke.name} 恢复了 {rise_hp} 点HP! 当前HP:{fight_poke.hp}/{fight_poke.max_hp}\n')

    sl(0.5)

    #电脑的技能
    skill2 = list(compu_fight_poke.skills.keys())#直接获取电脑当前出战宝可梦的技能
    #电脑的攻击

    #随机选择技能
    selected_skill_name2=skill2[random.randint(0,len(skill2)-1)]
    #计算战斗结果
    fin_damage,play_sta,sta_level,compu_sta,compu_sta_level,rise_hp=fight(compu_fight_poke,fight_poke,selected_skill_name2)
    #血量改变
    fight_poke.hp-=fin_damage
    compu_fight_poke.hp+=rise_hp
    #状态层数最多为10
    if compu_sta in sta_ty+de_sta_ty:
        compu_fight_poke.sta[compu_sta]=min(compu_fight_poke.sta[compu_sta]+sta_level,max_sta_level)
    if play_sta in sta_ty+de_sta_ty:
        fight_poke.sta[play_sta]=min(fight_poke.sta[play_sta]+sta_level,max_sta_level)
    #血量最低为0
    if fight_poke.hp<=0:
        fight_poke.hp=0
    #显示战斗结果
    if fin_damage>0:
        pr(f'\n你的 {fight_poke.name} 受到了 {fin_damage} 点伤害! 剩余HP:{fight_poke.hp}/{fight_poke.max_hp}\n')
    else:
        pr(f'\n你的 {fight_poke.name} 没有受到伤害!剩余HP:{fight_poke.hp}\n')
    if rise_hp>0:
        pr(f'\n敌方的 {compu_fight_poke.name} 恢复了 {rise_hp} 点HP! 当前HP:{compu_fight_poke.hp}/{compu_fight_poke.max_hp}\n')

    sl(0.5)
    pr('')

    #计算你的扣血状态
    del1,rise1=hp_del(fight_poke,compu_fight_poke)
    fight_poke.hp=max(fight_poke.hp-del1,0)
    compu_fight_poke.hp+=rise1
    pr('')
    if del1>0:
        pr(f'你的 {fight_poke.name} 减少了{del1}点生命!当前生命值:{fight_poke.hp}/{fight_poke.max_hp}\n')
    if rise1>0:
        pr(f'敌方的 {compu_fight_poke.name} 恢复了{rise1}点生命值!当前生命值:{compu_fight_poke.hp}/{compu_fight_poke.max_hp}\n')

    #计算电脑的扣血状态
    del2,rise2=hp_del(compu_fight_poke,fight_poke)
    compu_fight_poke.hp=max(compu_fight_poke.hp-del2,0)
    fight_poke.hp+=rise2
    pr('')
    if del2>0:
        pr(f'敌方的 {compu_fight_poke.name} 减少了{del2}点生命!当前的生命值:{compu_fight_poke.hp}/{compu_fight_poke.max_hp}\n')
    if rise2>0:
        pr(f'你的 {fight_poke.name} 恢复了{rise2}点生命值!当前生命值:{fight_poke.hp}/{fight_poke.max_hp}\n')
    
    #减少明状态层数
    sta_dec(compu_fight_poke,sta_ty)
    sta_dec(fight_poke,sta_ty)

    #结算宝可梦被击败
    if fight_poke.hp<=0:
        if isinstance(fight_poke, charmander):
            if fight_poke.fla_time > 0 and compu_fight_poke.hp>0:
                compu_fight_poke.mi_ra=max(compu_fight_poke.mi_ra-20*fight_poke.fla_time,0)
        if isinstance(fight_poke,bulbasaur):
            compu_fight_poke.sta['寄生']=0

        pr(f'\n{fight_poke.name} 被击败了!')
        sl(0.5)
        play.remove(fight_poke)
        fight_poke='wu'

    if compu_fight_poke.hp<=0:
        if isinstance(compu_fight_poke, charmander):
            if compu_fight_poke.fla_time > 0 and fight_poke.hp>0:
                fight_poke.mi_ra=max(fight_poke.mi_ra-20*compu_fight_poke.fla_time,0)
        if isinstance(compu_fight_poke,bulbasaur):
            fight_poke.sta['寄生']=0
        compu_fight_poke.hp=0
        pr(f'\n{compu_fight_poke.name} 被击败了!')
        sl(0.5)
        compu.remove(compu_fight_poke)
        compu_fight_poke='wu'
    
    #结算胜负
    if play==[] and compu==[]:
        pr('\n你和敌人的宝可梦全部战败!')
        sl(0.8)
        end1=0
        break
    if play==[]:
        pr('\n你的宝可梦全部战败!')
        sl(0.8)
        end1=1
        break
    if compu==[]:
        pr('\n敌人的宝可梦全部战败!')
        sl(0.8)
        end1=2
        break
    #text('战斗主循环')
    
    sl(0.5)

#结算
end_(end1)