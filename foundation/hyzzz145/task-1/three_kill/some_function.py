from card import Card,sha,equipment_card,scroll_card,Basic_card
from game import game
from player import Player



#定义展示卡牌函数
def show_card(card:Card):
    print(f"{type(card).__name__} [{card.suit}]",end="")
    if card.father != None:
        print(f"【由{card.father.suit}{type(card.father).__name__}转化】")
    else:
        print("")

#定义删除牌堆里卡牌的函数
def card_list_remove(card:Card):
    def remove(card):
        card_class = type(card).__bases__[0]
        if card_class == equipment_card:
            card_class = 0
        elif card_class == scroll_card:
            card_class = 1
        elif card_class == Basic_card:
            card_class = 2
        for i in game.card_list[card_class]:
                if i == card:
                    game.card_list[card_class].remove(i)
                    break
    remove(card)
    #判断是否为复制体
    if card.father != None:
        remove(card.father)

#展示单个玩家
def show_single_player(player:Player):
    print(f"玩家{player.id} 身份{player.identify},目前还剩生命值:{player.hp}/{player.max_hp}")
    
#展示所有玩家
def show_player():
    print("目前存活：")
    for i in game.player_list:
        show_single_player(i)
        
