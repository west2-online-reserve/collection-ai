#引用模块
from time import sleep;import random

#缩减写法
pr,inp,sl=print,input,sleep

max_sta_level=10

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
def create_ty(cls_name,ty_name):
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
Dian=create_ty('Dian','电')
Cao=create_ty('Cao', '草')
Shui=create_ty('Shui', '水')
Huo=create_ty('Huo', '火')
Beast=create_ty('Beast', '兽')
God=create_ty('God', '神')

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
        self.ascend_times=0
        self.skills={'1:飞升(修炼自己,提高了境界!\n获得一层“神化”状态(下一次受到的伤害减 10))':self.Ascendin,
                     f'2:神堕(释放愤怒,道心不稳!对敌方造成攻击力+10×使用过飞升次数点(当前为{10+10*self.ascend_times}点) 无视闪避的伤害,然后视为使用过的飞升次数清零)':self.Corruptin
        }
        

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
        # 每回合自动恢复5点月光能量
        if self.moon_energy < 100:
            self.moon_energy = min(self.moon_energy + 5, 100)
            pr(f'🌙 {self.name} 吸收了月光能量 (+5%)')
        pr(f'{self.name} 当前月光能量为:{self.moon_energy}%')
        pr(f'{self.name} 当前魅力值为:{self.charm_points}')

poke_list=[pikachu, bulbasaur, squirtle, charmander, fores_crow, god_huai, sevniao_rabbit]