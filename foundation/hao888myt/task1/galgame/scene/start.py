from character.jiejie import JieJie
from character.me import Me
from classes.util.dialogue_helper import (
    Choice,
    ChoiceGroup,
    Condition,
    DialogueExporter,
    DialogueGroup,
    Effect,
    EffectType,
)
from const.enum import ConditionType


def generator():
    DialogueExporter(
        [
            DialogueGroup(
                "group_1",
                [
                    "你是一名刚刚踏入大学校园的新生。",
                    "在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。",
                    "在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。",
                    "消息迅速传开，关于'神秘新生代表'的讨论充斥着整个校园。",
                    "于是，在这个新的舞台上，你与三位不同的女生产生了交集……",
                    "你回到教室，教室内的三位女生立刻引起了你的注意。",
                    "一位白色头发的女生趴在桌子上，旁边放着《Python入门》，似乎在为什么苦恼着。",
                    "一位学姐模样的女生坐在窗边，手里拿着画笔勾勒着什么。",
                    "后排的一名女生手里拿着一本书，专注地看着，完全没察觉到你的到来。",
                    "你决定上前找一位女生搭话。",
                ],
                "group_2",
            ),
            ChoiceGroup(
                "group_2",
                "你的选择是？",
                [
                    Choice("jie_jie", "哦天这是一个测试文本", "group_1"),
                    JieJie.choice("好感度够了", "group_1").add_condition(
                        Condition(ConditionType.REQUIRED_AFFINITY, 10)
                    ),
                    JieJie.choice("哦天这还是一个测试文本", "group_jiejie").add_effect(
                        Effect(EffectType.MODIFY_AFFINITY, 1)
                    ),
                ],
            ),
            DialogueGroup(
                "group_jiejie",
                [JieJie.talk("嗯？"), Me.talk("你好")],
                "group_2",
            ),
        ],
    )
