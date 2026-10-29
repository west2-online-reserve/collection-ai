from card import  (
    Basic_card,
    Sha,
    Tao,
    Shan,
    Guohechaiqiao,
    Shunshouqianyang,
    Wuzhongshengyou,
    Nanmanruqin,
    Juedou,
    Lebusishu,
    Wuxiekeji,
    Shandian,
    Wugufengdeng,
    Jiedaosharen,
    Wanjianqifa,
    Taoyuanjieyi,
    Huogong,
    Tiesuolianhuan,
    Bingliangcunduan,
)
from effect import Effect

eff = Effect()
#定义 game
class Game():
    card_list = []
    player_list = []
    discard_list = []
    def __init__(self):
        self.run = 1
        self.start(self)
        self.card_list = []
        self.player_list = []
        self.discard_list = []
    
    
    def start(self):
        #基础牌初始化
        self.card_list += (
        #初始化杀
        [Sha("黑桃") for _ in range(7)] + [Sha("梅花") for _ in range(14)] + [Sha("红桃") for _ in range(3)] + [Sha("方块" for _ in range(6))] +
        #桃
        [Tao("红桃") for _ in range(7)] + [Tao("方块") for _ in range(1)] +
        #闪
        [Shan("红桃") for _ in range(3)] + [Shan("方块") for _ in range(12)] +
        # 过河拆桥：黑桃3张、梅花2张、红桃1张
        [Guohechaiqiao("黑桃") for _ in range(3)] + [Guohechaiqiao("梅花") for _ in range(2)] + [Guohechaiqiao("红桃") for _ in range(1)] +

        # 顺手牵羊：黑桃3张、方块2张
        [Shunshouqianyang("黑桃") for _ in range(3)] + [Shunshouqianyang("方块") for _ in range(2)] +

        # 无中生有：红桃4张
        [Wuzhongshengyou("红桃") for _ in range(4)] +

        # 南蛮入侵：黑桃2张、梅花1张
        [Nanmanruqin("黑桃") for _ in range(2)] + [Nanmanruqin("梅花") for _ in range(1)] +

        # 决斗：黑桃1张、梅花1张、方块1张
        [Juedou("黑桃") for _ in range(1)] + [Juedou("梅花") for _ in range(1)] + [Juedou("方块") for _ in range(1)] +

        # 乐不思蜀：黑桃1张、红桃1张、梅花1张
        [Lebusishu("黑桃") for _ in range(1)] + [Lebusishu("红桃") for _ in range(1)] + [Lebusishu("梅花") for _ in range(1)] +

        # 无懈可击：黑桃2张、红桃2张、梅花2张、方块1张
        [Wuxiekeji("黑桃") for _ in range(2)] + [Wuxiekeji("红桃") for _ in range(2)] + [Wuxiekeji("梅花") for _ in range(2)] + [Wuxiekeji("方块") for _ in range(1)] +

        # 闪电：黑桃1张、红桃1张
        [Shandian("黑桃") for _ in range(1)] + [Shandian("红桃") for _ in range(1)] +

        # 五谷丰登：红桃2张
        [Wugufengdeng("红桃") for _ in range(2)] +

        # 借刀杀人：梅花2张
        [Jiedaosharen("梅花") for _ in range(2)] +

        # 万箭齐发：红桃1张
        [Wanjianqifa("红桃") for _ in range(1)] +

        # 桃园结义：红桃1张
        [Taoyuanjieyi("红桃") for _ in range(1)] +

        # 火攻：红桃2张、方块1张
        [Huogong("红桃") for _ in range(2)] + [Huogong("方块") for _ in range(1)] +

        # 铁索连环：黑桃2张、梅花4张
        [Tiesuolianhuan("黑桃") for _ in range(2)] + [Tiesuolianhuan("梅花") for _ in range(4)] +

        # 兵粮寸断：黑桃1张、梅花1张
        [Bingliangcunduan("黑桃") for _ in range(1)] + [Bingliangcunduan("梅花") for _ in range(1)]

        )
        
                    
        self.card_list

    #状态更新
    def updata(self):
        #玩家是否死亡
        for i in self.player_list:
            if i.hp <= 0:#111
                i.near_dead = True
                while(i.near_dead):
                    if eff.force_tao(i):
                        i.near_dead = False
                    else:
                        i.near_dead = False
                        i.dead = True
            if i.dead:
                self.player_list.remove(i)
                        


                
    #定义返回特定 id 玩家的函数
    def return_player(self):
        for i in self.player_list:
            if i.id == self.turn_id:
                return i 
    
    #阶段一
    def stage_one(self):
        if self.return_player(self).rule == None: 
            #stage_one_0 
            for i in self.return_player(self).one:
                i.act()
                self.return_player(self).one.remove(i)
                #updata(self)



    #规则修改型事件；定义 rule ，一般rule = '1' 就是没修改的状态，rule = '特定字符' 进入特定阶段函数，阶段跳过是一种特殊的规则修改
    def rule_updata(self):
        pass

    #卡牌转换型事件；检验特定牌组，若纯在，则添加一个复制型的牌卡，此牌卡拥有特定父级属性（father:card），即在打出父级不为None的卡牌时，自动删去其父级
    #状态标记；对此玩家添加一个状态属性，对拥有此属性的玩家调用特定函数
    #主动技能；在准备阶段，供玩家选择
    #事件监听；在某件事前面检测是否触发这个事件发生条件，若发生，修改事件的特定属性
    def game_looping(self):
        #游戏初始化
        self.run = 1
        self.turn_id = 0
         #游戏流程循环
        while self.run:
    #定义变量 turn_id
            self.turn_id = (self.turn_id)%len() + 1
    #阶段一
        
    #阶段二
    #阶段三
if __name__ == "__main__":
    game = Game
    game.start()
    
    
    