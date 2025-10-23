#引用模块
import random
from time import sleep
from pokemon_cls import (
    poke_list, sta_ty, de_sta_ty, ty_list, max_sta_level,
    Pokenmon, pikachu, bulbasaur, squirtle, charmander, 
    fores_crow, god_huai, sevniao_rabbit
)


#缩减写法
pr,inp,sl=print,input,sleep


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
    play=[];compu=[]    
    compu_ch=random.sample(range(1,len(poke_list)+1),3)
    compu_cho=[str(i) for i in compu_ch]

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

#选择出战宝可梦
def cho_fight_poke(fight_poke,compu_fight_poke,play,compu):
    if fight_poke=='wu':
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
        compu_fight_poke=compu[random.randint(0,len(compu)-1)]
        pr(f'\n敌人选择了 {compu_fight_poke.name}')

    return fight_poke,compu_fight_poke

#回合开始函数
def turn_start(fight_poke,compu_fight_poke,cur_turn):
    new_turn=cur_turn+1
    sl(0.5)

    #回合开始
    pr(f'\n--------回合{new_turn}--------')

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
    
    return new_turn

#选择技能函数
def cho_skill(fight_poke):
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

    #选择技能
    move1=int(move1)
    selected_skill_name1=skill1[move1-1]
    return selected_skill_name1

#结算战斗函数
def fight_change(me,enemy,fin_damage,rise_hp,enemy_sta,enemy_sta_level,my_sta,my_sta_level):
    #血量改变
    enemy.hp=max(0,enemy.hp-fin_damage)
    me.hp=min(me.max_hp,me.hp+rise_hp)
    #状态层数最多为10
    if enemy_sta in sta_ty+de_sta_ty:
        enemy.sta[enemy_sta]=min(enemy.sta[enemy_sta]+enemy_sta_level,max_sta_level)
    if my_sta in sta_ty+de_sta_ty:
        me.sta[my_sta]=min(me.sta[my_sta]+my_sta_level,max_sta_level)

#展示战斗结果函数
def show_fight(me,enemy,damage,rise_,contr_name,oppon_name):
    if damage>0:
        pr(f'\n{oppon_name} {enemy.name} 受到了 {damage} 点伤害! 剩余HP:{enemy.hp}/{enemy.max_hp}\n')
    else:
        pr(f'\n{oppon_name} {enemy.name} 没有受到伤害!剩余HP:{enemy.hp}\n')
    if rise_>0:
        pr(f'\n{contr_name} {me.name} 恢复了 {rise_} 点HP! 当前HP:{me.hp}/{me.max_hp}\n')
    sl(0.5)
    pr('')

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
def hp_del(me,enemy,contr_name,oppon_name):
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
        enemy_rise=jisheng_rise
        total_decl+=jisheng_decl
    if me.sta['天诛']>0:
        total_decl=me.hp
        pr(f'{me.name} 被天诛!生命值归零')
    
    me.hp=max(me.hp-total_decl,0)
    enemy.hp=min(enemy.max_hp,enemy.hp+enemy_rise)
    pr('')

    if total_decl>0:
        pr(f'{contr_name} {me.name} 减少了{total_decl}点生命!当前生命值:{me.hp}/{me.max_hp}\n')
    if enemy_rise>0:
        pr(f'{oppon_name} {enemy.name} 恢复了{enemy_rise}点生命值!当前生命值:{enemy.hp}/{enemy.max_hp}\n')

#结算宝可梦战败
def poke_lose(me,enemy,team_list,contr_name):
    if me.hp<=0:
        if isinstance(me,charmander):
            if me.fla_time > 0 and enemy.hp>0:
                enemy.mi_ra=max(enemy.mi_ra-20*me.fla_time,0)
        if isinstance(me,bulbasaur):
            enemy.sta['寄生']=0

        pr(f'\n{contr_name} {me.name} 被击败了!')
        sl(0.5)
        team_list.remove(me)
        me='wu'
    return me
    
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

#结算胜负函数
def if_end(play,compu,end1,gameover=False):
    if play==[] and compu==[]:
        pr('\n你和敌人的宝可梦全部战败!')
        sl(0.8)
        end1=0
    elif play==[]:
        pr('\n你的宝可梦全部战败!')
        sl(0.8)
        end1=1
    elif compu==[]:
        pr('\n敌人的宝可梦全部战败!')
        sl(0.8)
        end1=2

    if end1 in [x for x in range(3)]:
        gameover=True
        
    return gameover,end1

#结束函数
def end_(end1):
    if end1==1:
        pr('\n在敌人的猛烈攻势下,你的宝可梦完全不敌,战败了...')
    elif end1==2:
        pr('\n你和你的宝可梦齐心协力,打倒了敌人!')
    elif end1==3:
        pr('\n出乎意料,你们几乎势均力敌,最终不分胜负!')
    elif end1==4:
        pr('\n测试结束!')
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

