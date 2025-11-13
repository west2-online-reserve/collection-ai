'''游戏主程序'''

from typing import Generator,List,Dict
from story import *
import sys
import time

class Game:
    '''实现游戏IO'''
    def __init__(self) -> None:
        '''初始化游戏'''
        self.stories = Story()
        self.story_script:Generator[List[str | dict], str | None, StopIteration] = self.stories.build_story()
        self.ans: str | None = None
        self.question: bool = False
        self.name: str | None = None
        self.text_list: List[str | dict] = None

    def print_f(self,text_list: List[str | dict]) -> None:
        '''格式化打印剧情文本'''
        for state in text_list:
            #输出普通文本
            if isinstance(state,str):
                #time.sleep(1)
                print(state)
            #输出选项文本
            elif isinstance(state,dict):
                if state.get("text"):
                    print(f"{self.name}：『{state['text']}』")
                    #time.sleep(1)
                    print("\n你的回答是？")
                    print(f"选项A：{state['a']}")
                    print(f"选项B：{state['b']}")
                    self.question = True
                elif state.get("entry") in ("choice","gift"):
                    self.question = True
                elif state["entry"] == "quit":
                    sys.exit(0)

    def start(self) -> None:
        '''游戏主循环，接收来自Story的剧情文本'''
        while True:
            try:
                self.text_list = self.story_script.send(self.ans)
            except StopIteration :
                sys.exit(0)

            self.name = self.stories.current_character
            self.print_f(self.text_list)

            #判断当前状态是否需要玩家输入
            if self.question:
                self.ans = input("请输入你的选择：").strip().lower()
            else:
                self.ans = None
            self.question = False

if __name__ == "__main__":
    '''游戏开始'''
    game=Game()
    game.start()