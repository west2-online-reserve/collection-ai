from __future__ import annotations
import subprocess
import platform 
import story
from story import Character
from manage import slow_print
from manage import Game_manage


def clear_screen() -> None:
    '''清屏'''
    if platform.system() == 'Windows':
        subprocess.run('cls', shell = True)
    else:
        subprocess.run('clear', shell = True)


class Game():
    def __init__(self):
        self.characters: list[Character] = [cls() for cls in Character.__subclasses__()]
        self.current_target: Character = None
        self.manage = Game_manage(self)

    def scene_talk(self, target: Character, count: int) -> None:
            scene = story.DIALOGUES.get(target.name)[count]
            slow_print(text = scene['text'])
            slow_print(text = scene['optionA'])
            slow_print(text = scene['optionB'])

    def start(self) -> None:
        '''游戏开始时'''
        clear_screen()
        print("========== 游戏开始：校园 if·恋 ==========")
        slow_print("你是一名刚刚踏入大学校园的新生。")
        slow_print("在开学典礼上，拿下压倒性成绩第一的你被选为新生代表发言。")
        slow_print("在全场上千人的注视下，你气质非凡，发言流畅，很快成为焦点人物。")
        slow_print("消息迅速传开，关于‘神秘新生代表’的讨论充斥着整个校园。")
        slow_print("于是，在这个新的舞台上，你与三位不同的女生产生了交集……")

    def choose(self) -> bool:
        '''选择女主'''
        chose = 0
        for i in self.characters:
            self.current_target = i
            scene = story.DIALOGUES[i.name][0]
            self.manage.talk(scene = scene)
            if i.affinity <= 0:
                continue
            else:
                chose = 1
                i.affinity = 0
                break
        if not chose:
            print('\n啥，眼前三妹子都不要？？死现充别玩 galgame')
            return False
        return True

    def daily(self) -> None:
        '''主循环'''
        for i in range(1,5):
            scene = story.DIALOGUES[self.current_target.name][i]
            self.manage.opt(scene = scene)
            if self.current_target.affinity >= 50 and not self.current_target.petname:
                self.current_target.pick_petname()
            if i == 4:
                if self.current_target.affinity <= 50:
                    slow_print(f'你和{self.current_target.name}的故事就这样结束了')
                elif self.current_target.affinity <= 100:
                    slow_print(f'你和{self.current_target.name}未能修成正果')
                else:
                    slow_print(self.current_target.prefect_end())

    def run(self) -> None:
        '''运行'''
        self.start()
        if self.choose():
            self.daily()


if __name__ == '__main__':
    galgame = Game()
    galgame.run()

