from __future__ import annotations
import story
import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import Game



def slow_print(text: str, speed: float = 0.05) -> None:
    '''慢输出'''
    for i in text:
        print(i, end = '', flush = True)
        time.sleep(speed)
    print()


class Game_manage():
    def __init__(self, game):
        self.game: Game = game

    def opt(self, scene: dict[str, str]):
        '''进行大选择'''
        print(f'面对{self.game.current_target.name}，你要做：')
        while(1):
            print('A. 与她对话')
            print('B. 查看她的当前信息')
            print('C. 向她送出你准备的礼物')
            option = input('你选择：')
            match option:
                case 'A':
                    self.talk(scene)
                    self.game.current_target.have_gift = 0
                    break
                case 'B':
                    self.view()
                case 'C':
                    if self.game.current_target.have_gift:
                        print('今日送礼物次数到达上限。')
                    else:
                        self.game.current_target.have_gift += 1
                        self.selection_gift()

    def talk(self, scene: dict[str, str]) -> None:
        '''小选择：对话'''
        slow_print(f'{self.game.current_target.name}向你走来')
        slow_print(f'「{scene['text']}」')
        print(f'A.「{scene['optionA']}」')
        print(f'B.「{scene['optionB']}」')
        option = input('你准备对她说：')
        while(1):
            if option == 'A':
                self.game.current_target.change_affinity(value = 30)
                break
            elif(option == 'B'):
                self.game.current_target.change_affinity(value = -10)
                break
            else:
                option = input('你应该从 A,B 中选择一个进行回答')
                

    def view(self) -> None:
        '''打印信息'''
        print(self.game.current_target.name)
        if self.game.current_target.petname:
            print(f'（{self.game.current_target.petname}）')
        print(f'身份：{self.game.current_target.identity}')
        print(f'外貌：{self.game.current_target.appearance}')
        print(f'性格：{self.game.current_target.personality}')
        print(f'好感度：{self.game.current_target.affinity}')

    def selection_gift(self) -> None:
        '''小选择：礼物'''
        print(f'请为{self.game.current_target.name}挑选礼物')
        for i in story.GIFT_EFFECTS.keys():
            print(f'{i}', end = ' ')
        gift = input('你选择：（要返回的话，请输入“返回”或“ return ”）')
        while(1):
            try:
                self.game.current_target.gift_effect(gift)
                break
            except:
                if gift.strip().lower() in ['return', '返回']:
                    break
                else:
                    print('请正确输入礼物的正确名称')
                    continue