"""游戏流程：开场、三次出场选择与角色线主循环。"""

from __future__ import annotations

from characters import Character
from story import (
    BROKEN_ENDING,
    ENDINGS,
    ENDING_AFFINITY,
    GIFT_EFFECTS,
    NO_ROUTE_ENDING,
    PROLOGUE,
    SCENES,
)
from ui import ask_choice, divider


class Game:
    """一局游戏的全部状态。"""

    def __init__(self) -> None:
        self.characters: dict[str, Character] = {
            "学姐": Character("学姐", "社团里的艺术少女"),
            "小白": Character("小白", "课堂上的元气同学"),
            "姐姐": Character("姐姐", "食堂里的温柔姐姐"),
            "None": Character("None", "无"),
        }
        self.current_target: Character = self.characters["None"]
        self.result: str = ""

    # ------------------------------------------------------------ 开场

    def start(self) -> None:
        """开始并跑完一局游戏。"""
        self.show_prologue()

        # 依次进入三个场景；一旦选定路线，后面的场景自动跳过。
        reached = self.scene_senpai() or self.scene_xiaobai() or self.scene_jiejie()
        if not reached:
            print("\n啥，眼前三妹子都不要？？死现充别玩 galgame")
            self._show_ending_lines(NO_ROUTE_ENDING)

        print("\n感谢游玩《校园·if 恋》。")

    def show_prologue(self) -> None:
        """打印开场旁白。"""
        divider("序章")
        for line in PROLOGUE:
            print(line)

    # ------------------------------------------------------------ 出场场景

    def scene_senpai(self) -> bool:
        """场景一：学姐。"""
        return self._run_scene("senpai")

    def scene_xiaobai(self) -> bool:
        """场景二：小白。"""
        return self._run_scene("xiaobai")

    def scene_jiejie(self) -> bool:
        """场景三：姐姐。"""
        return self._run_scene("jiejie")

    def _run_scene(self, key: str) -> bool:
        """通用场景流程：介绍 -> 二选一 -> 进入角色线或错过。"""
        scene = SCENES[key]
        divider(scene["title"])
        for line in scene["intro"]:
            print(line)

        choice = ask_choice(scene["options"], {"1", "2"})
        print()
        outcome = scene["accept"] if choice == "1" else scene["reject"]
        for line in outcome:
            print(line)

        if choice == "1":
            self.current_target = self.characters[scene["heroine"]]
            self.story_loop()
            return True
        return False

    # ------------------------------------------------------------ 主循环

    def story_loop(self) -> str:
        """角色线主循环，返回结束原因：ending / broken / left。"""
        target = self.current_target
        self._show_route_header()

        while True:
            print("\n你要做什么？")
            print("1. 和她聊天")
            print("2. 送她礼物")
            print("3. 查看好感度")
            print("4. 离开（退出游戏）")

            choice = ask_choice("请输入选项：", {"1", "2", "3", "4"})

            if choice == "4":
                print(f"你向{target.name}道别，转身离开。")
                print("游戏结束。")
                self.result = "left"
                return self.result

            if choice == "1":
                target.talk()
            elif choice == "2":
                self._do_gift()
            else:
                self.show_affinity()

            if target.check_ending():
                self.show_ending()
                self.result = "ending"
                return self.result
            if target.is_broken():
                self._show_ending_lines(BROKEN_ENDING)
                self.result = "broken"
                return self.result

    def _do_gift(self) -> None:
        """让玩家从礼物列表中选择一件送给当前角色。"""
        gifts = list(GIFT_EFFECTS)
        print("\n要送她什么？")
        for index, gift in enumerate(gifts, start=1):
            print(f"{index}. {gift}")
        print("0. 返回")

        choice = ask_choice("请选择：", {str(i) for i in range(len(gifts) + 1)})
        if choice == "0":
            return
        self.current_target.give_gift(gifts[int(choice) - 1])

    # ------------------------------------------------------------ 展示

    def _show_route_header(self) -> None:
        """打印当前路线与好感度。"""
        target = self.current_target
        divider(f"{target.name}线")
        print(f"角色：{target.role}")
        print(f"好感度：{target.affinity}/{ENDING_AFFINITY}")

    def show_affinity(self) -> None:
        """查看当前好感度详情。"""
        target = self.current_target
        remaining = ENDING_AFFINITY - target.affinity
        print(f"\n【{target.name}】好感度：{target.affinity}/{ENDING_AFFINITY}")
        print(f"[{self._affinity_bar(target.affinity)}]")
        print(f"已经聊过 {target.talk_count} 次，送过 {target.gift_count} 件礼物。")

        if target.affinity >= ENDING_AFFINITY:
            print("心意已经传达到了。")
        elif target.affinity >= 80:
            print(f"她看你的眼神已经有些不一样了。还差 {remaining} 点就能走到结局。")
        elif target.affinity >= 40:
            print(f"关系正在升温，再加把劲吧（还差 {remaining} 点）。")
        elif target.affinity >= 0:
            print(f"还需要更多相处（还差 {remaining} 点）。")
        else:
            print("气氛有点冷，再这样下去她可能会离开。")

    @staticmethod
    def _affinity_bar(affinity: int) -> str:
        """用 10 格字符画一个好感度进度条。"""
        filled = max(0, min(10, affinity // 10))
        return "■" * filled + "□" * (10 - filled)

    def show_ending(self) -> None:
        """播放当前角色的结局。"""
        target = self.current_target
        divider("结局")
        for line in ENDINGS[target.name]:
            print(line)
        print(f"\n最终好感度：{target.affinity}/{ENDING_AFFINITY}")
        print("—— 全剧终 ——")

    def _show_ending_lines(self, lines: tuple[str, ...]) -> None:
        """播放一段通用结局文字。"""
        divider("结局")
        for line in lines:
            print(line)
        print("—— 全剧终 ——")
