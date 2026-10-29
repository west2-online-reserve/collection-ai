from player import Player
from some_function import show_card

#效果类
class Effect():
    def __init__(self):
        pass
    #def 
    #杀
    def sha(self,opponent:Player):
        #判断四名玩家是否出无懈可击

        self.reduce_hp(self,1,opponent)
        #判断对方是否出闪，出了则调用闪效果

    #桃子
    def tao(self):
        self.recover_hp(self,1)
    #强制出桃
    def force_tao(self,oppent:Player):
        idx = 0
        #储存桃
        tao = []
        for i in oppent.card_list[2]:
            if type(i) == 'tao':
                idx += 1
                print(f"{idx}.",end="")
                show_card(f"{i} ")
                tao.append(i)
        #没有桃的处理情况
        if len(tao) == 0:
            print(f"玩家{oppent.id}没有桃")
            return False
        else:
            choice = int(input("选择一张桃,或者放弃选择（选择-1）"))
            #数据溢出处理
            if choice > len(tao) and choice <= 0:
                raise  ValueError(f"请属于位于(1-{len(tao)}或者-1的数)")
            if choice == -1:
                return False
            else:
                print("使用了",end="")
                show_card(tao[choice - 1])
                tao(self)
                return True


        
        
                


    #无懈可击
        #玩家
        #有无懈可击，并且选择出，删除这个牌并返回 ture；无或者不出直接返回 false

        #人机
        #有无懈可击，判断 opponent 是不是自己，是则出；不是则不出或者没有无懈可击
    #闪（判断是否有闪，并决定出不出）

        #玩家
        #有闪，并且选择出，删除这个闪并返回 ture；无闪或者不出直接返回 false

        #人机
        #有闪，删除并返回 ture，无闪返回 return

    #借刀杀人
        #判断对手是否装备区有武器牌，若无则重新选择对手
        #有的话，强制对手出杀，并让对手选择一个非自己的角色
        #若不出的话，将武器牌给你

    #过河拆桥
        #弃掉对方三个区域的一张牌

    #决斗
        #轮流打杀，创造一个循环，谁没杀时则让这个函数记录谁的 id，并使其造成一点伤害

    #闪电_放置
        #放入你的 three 列表
    #闪电
        #若抽到黑心 2-9 的牌，触发闪电判定，否则继续调用【闪电放置】，目标为（判定者 id + 1）% 4 的玩家

    #火攻
        #先让对方展示手牌，人机随机，真人自选
        #如果你存在这种同样花色的牌，可以选择展示或者不展示，人机默认展示；展示后对目标造成 1 点火焰伤害
    
    #乐不思蜀_放置
        #对目标使用，若目标判定不为红桃，则跳过其回合

    #英雄技能
    #张飞
        #咆哮：无限使用杀，即修改规则，把rule改为“zhangfei”
    #赵云
        #牌卡转换，杀当闪，闪当杀
    #关羽
        #武圣：牌卡转换，把红色牌打出
    #吕蒙
        #修改规则+事件监听，回合内未打出杀，则跳过弃牌阶段
    #孙权
        #修改规则，一个回合，可以弃任意张牌，再抽这么多牌
    #曹操
        #事件监听（对别人），对自己造成伤害的牌加入自己卡组
    #黄月英
        #事件监听，使用锦囊牌，摸一张
    #甘宁
        #牌卡转换；黑色牌当过河拆桥