"""角色类：负责一名女主角的好感度、对话与送礼逻辑。"""

from __future__ import annotations

from story import (
    BOND_AFFINITY,
    BOND_DIALOGUES,
    BROKEN_AFFINITY,
    DIALOGUES,
    ENDING_AFFINITY,
    GIFT_COMMENTS,
    GIFT_EFFECTS,
    DialogueNode,
)
from ui import ask_choice


class Character:
    """一名可攻略角色。"""

    def __init__(self, name: str, role: str, affinity: int = 0) -> None:
        self.name: str = name
        self.role: str = role
        self.affinity: int = affinity
        self.talk_count: int = 0
        self.gift_count: int = 0
        self.bond_seen: bool = False

    # ------------------------------------------------------------ 对话

    def talk(self) -> int:
        """进行一次对话，返回本次好感度变化值。"""
        print(f"\n你正在和{self.name}聊天……")
        node, repeated = self._pick_dialogue()
        if repeated:
            print("（这个话题你们之前聊过，她的反应平淡了些。）")
        print(f"{self.name}：『{node['prompt']}』")
        print(f"A. {node['choices']['A']['text']}")
        print(f"B. {node['choices']['B']['text']}")

        key = ask_choice("请选择（A/B）：", {"A", "B"})
        chosen = node["choices"][key]
        print(f"你：『{chosen['text']}』")
        print(f"{self.name}：『{chosen['reply']}』")

        gain = chosen["affinity"]
        if repeated:
            # 重复话题收益减半，避免只靠一句话反复刷满好感度。
            gain = gain // 2 if gain > 0 else -((-gain) // 2)
            if chosen["affinity"] > 0:
                gain = max(gain, 1)

        self.talk_count += 1
        self.change_affinity(gain)
        return gain

    def _pick_dialogue(self) -> tuple[DialogueNode, bool]:
        """挑选本次对话节点，并返回 (节点, 是否为重复话题)。"""
        if self.affinity >= BOND_AFFINITY and not self.bond_seen:
            self.bond_seen = True
            return BOND_DIALOGUES[self.name], False

        nodes = DIALOGUES[self.name]
        repeated = self.talk_count >= len(nodes)
        return nodes[self.talk_count % len(nodes)], repeated

    # ------------------------------------------------------------ 送礼

    def give_gift(self, gift: str) -> int:
        """送出礼物，返回本次好感度变化值。"""
        print(f"\n你送给{self.name}一份「{gift}」。")
        effects = GIFT_EFFECTS.get(gift)
        if effects is None:
            print(f"{self.name}：『……这是什么？』")
            return 0

        delta = effects.get(self.name, effects.get("default", 0))
        comment = (
            GIFT_COMMENTS.get(self.name, {}).get(gift)
            or GIFT_COMMENTS["*"].get(gift)
        )
        if comment:
            print(f"{self.name}：{comment}")
        else:
            print(f"{self.name}：『{self._gift_reply(delta)}』")

        self.gift_count += 1
        self.change_affinity(delta)
        return delta

    @staticmethod
    def _gift_reply(delta: int) -> str:
        """根据好感度变化返回一句通用回应。"""
        if delta >= 15:
            return "这个……我很喜欢，谢谢你。"
        if delta >= 5:
            return "谢谢你，我会好好收着的。"
        if delta > 0:
            return "谢谢。"
        if delta == 0:
            return "谢谢，不过这个我大概用不上。"
        return "……你确定这是送给我的吗？"

    # ------------------------------------------------------------ 数值

    def change_affinity(self, value: int) -> None:
        """修改好感度并打印变化（上限为结局阈值）。"""
        before = self.affinity
        self.affinity = max(-100, min(ENDING_AFFINITY, before + value))
        actual = self.affinity - before
        print(f"{self.name}的好感度 {actual:+d} -> 当前好感度：{self.affinity}")

    def check_ending(self) -> bool:
        """好感度是否已达到结局线。"""
        return self.affinity >= ENDING_AFFINITY

    def is_broken(self) -> bool:
        """好感度是否已低到让角色离开。"""
        return self.affinity <= BROKEN_AFFINITY
