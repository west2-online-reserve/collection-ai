#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#引入模块
from time import sleep;import random;from sys import exit
from pokemon_cls import (
    poke_list, sta_ty, de_sta_ty, ty_list, max_sta_level,
    Pokenmon, pikachu, bulbasaur, squirtle, charmander, 
    fores_crow, god_huai, sevniao_rabbit
)
from pokemon_funct import (
    check_poke,cho_poke,cho_fight_poke,turn_start,
    cho_skill,fight,fight_change,show_fight,hp_del,sta_dec,poke_lose,if_end
    ,end_,text
)

#缩减写法
pr,inp,sl=print,input,sleep

#一些提前写好的变量
play_name='你的';compu_name='敌方的'
fight_poke='wu';compu_fight_poke='wu'
turn=0
end1=4

#欢迎
pr('欢迎来到文字宝可梦!')

#检查宝可梦
check_opt=inp('你需要查看宝可梦图鉴吗?\n若需要,请输入 c (不需要请直接按下回车键):')
if check_opt=='c':
    check_poke()


#正式开始

#选宝可梦
play,compu=cho_poke()

#核心战斗主循环
while True:

    #选择出战宝可梦
    fight_poke,compu_fight_poke=cho_fight_poke(fight_poke,compu_fight_poke,play,compu)

    #开始战斗

    #回合开始
    turn=turn_start(fight_poke,compu_fight_poke,turn)
    
    #你的攻击

    #选择技能
    selected_skill_name1=cho_skill(fight_poke)
    #战斗
    fin_damage,compu_sta,compu_sta_level,play_sta,sta_level,rise_hp=fight(fight_poke, compu_fight_poke, selected_skill_name1)
    #结算
    fight_change(fight_poke,compu_fight_poke,fin_damage,rise_hp,compu_sta,compu_sta_level,play_sta,sta_level)   
    #显示战斗结果
    show_fight(fight_poke,compu_fight_poke,fin_damage,rise_hp,play_name,compu_name)

    #电脑的攻击

    #电脑的技能
    skill2 = list(compu_fight_poke.skills.keys())#直接获取电脑当前出战宝可梦的技能
    #随机选择技能
    selected_skill_name2=skill2[random.randint(0,len(skill2)-1)]
    #战斗
    fin_damage,play_sta,sta_level,compu_sta,compu_sta_level,rise_hp=fight(compu_fight_poke,fight_poke,selected_skill_name2)
    #结算
    fight_change(compu_fight_poke,fight_poke,fin_damage,rise_hp,play_sta,sta_level,compu_sta,compu_sta_level)
    #显示战斗结果
    show_fight(compu_fight_poke,fight_poke,fin_damage,rise_hp,compu_name,play_name)


    #计算并展示你的扣血状态
    hp_del(fight_poke,compu_fight_poke,play_name,compu_name)

    #计算并展示电脑的扣血状态
    hp_del(compu_fight_poke,fight_poke,compu_name,play_name)
    
    #减少明状态层数
    sta_dec(compu_fight_poke,sta_ty)
    sta_dec(fight_poke,sta_ty)

    #结算宝可梦被击败
    fight_poke=poke_lose(fight_poke,compu_fight_poke,play,play_name)
    compu_fight_poke=poke_lose(compu_fight_poke,fight_poke,compu,compu_name)
    
    #结算胜负
    gameover,end1=if_end(play,compu,end1)
    if gameover:
        break
    #text('战斗主循环')
    
    sl(0.5)

#结算
end_(end1)