from effect import Effect
#卡牌主类
class Card():
    def __init__(self,suit:str,effect:Effect)->None:
        #花色
        self.suit = suit
        #触发时间，0代表现在
        #self.act_time = act_time
        self.father = None

#基本牌
class Basic_card(Card):
    def __init__(self, suit,effect):
        super().__init__(suit,effect)

class Sha(Basic_card):
    def __init__(self, suit):
        super().__init__(suit, Effect.sha(self))

class Tao(Basic_card):
    pass

class Shan(Basic_card):
    pass

#锦囊牌
class Scroll_card():
    pass
# 过河拆桥
class Guohechaiqiao(Scroll_card):
    pass

# 顺手牵羊
class Shunshouqianyang(Scroll_card):
    pass

# 无中生有
class Wuzhongshengyou(Scroll_card):
    pass

# 南蛮入侵
class Nanmanruqin(Scroll_card):
    pass

# 决斗
class Juedou(Scroll_card):
    pass

# 乐不思蜀
class Lebusishu(Scroll_card):
    pass

# 无懈可击
class Wuxiekeji(Scroll_card):
    pass

# 闪电
class Shandian(Scroll_card):
    pass

# 五谷丰登
class Wugufengdeng(Scroll_card):
    pass

# 借刀杀人
class Jiedaosharen(Scroll_card):
    pass

# 万箭齐发
class Wanjianqifa(Scroll_card):
    pass

# 桃园结义
class Taoyuanjieyi(Scroll_card):
    pass

# 火攻
class Huogong(Scroll_card):
    pass

# 铁索连环
class Tiesuolianhuan(Scroll_card):
    pass

# 兵粮寸断
class Bingliangcunduan(Scroll_card):
    pass

#装备牌
class Equipment_card():
    pass 

#诸葛连弩
class Zhugeliannu(Equipment_card):
    pass
# 雌雄双股剑
class Cixiongshuanggujian(Equipment_card):
    pass

# 青釭剑
class Qinggangjian(Equipment_card):
    pass

# 丈八蛇矛
class Zhangbashemao(Equipment_card):
    pass

# 贯石斧
class Guanshifu(Equipment_card):
    pass

# 青龙偃月刀
class Qinglongyanyuedao(Equipment_card):
    pass

# 方天画戟
class Fangtianhuaji(Equipment_card):
    pass

# 麒麟弓
class QilinGong(Equipment_card):
    pass

# 寒冰剑
class Hanbingjian(Equipment_card):
    pass

# 古锭刀
class Gudingdao(Equipment_card):
    pass

# 朱雀羽扇
class Zhuqueyushan(Equipment_card):
    pass

# 八卦阵
class Baguazhen(Equipment_card):
    pass

# 仁王盾
class Renwangdun(Equipment_card):
    pass

# 藤甲
class Tengjia(Equipment_card):
    pass

# 白银狮子
class Baiyinshizi(Equipment_card):
    pass

# 绝影
class Jueying(Equipment_card):
    pass

# 的卢
class Dilu(Equipment_card):
    pass

# 爪黄飞电
class Zhuahuangfeidian(Equipment_card):
    pass

# 赤兔
class Chitu(Equipment_card):
    pass

# 大宛
class Dayuan(Equipment_card):
    pass

# 紫骍
class Zixing(Equipment_card):
    pass

# 骅骝
class Hualiu(Equipment_card):
    pass
