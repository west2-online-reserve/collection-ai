'''生成剧情文本'''

from typing import Generator,List,Dict
from manage import Manage
from scripts import SPECIAL,PROLOGUE,SENPAI,XIAOBAI,JIEJIE
import sys

class Story:
    '''生成剧情文本'''
    def __init__(self) -> None:
        '''初始化'''
        self.manager = Manage()
        self.current_route: str = "p"
        self.characters: dict = {"s":"学姐","x":"小白","j":"姐姐"}
        self.current_character: str | None = None

        self.current_node: int | str = 0
        self.current_question: dict= None
        self.special: str = None
    
    def script(self,route: str, node: int | str, special:str) -> List[str | dict]:
        '''选取剧情文本'''
        if special == None:
            if isinstance(node,int):
                node=str(node).zfill(2)
                if route+node == "p01":
                    self.current_character="学姐"
                elif route+node == "p03":
                    self.current_character="小白"
                elif route+node == "p05":
                    self.current_character="姐姐"

            if route == "p": return PROLOGUE[route+node]
            elif route == "s": return SENPAI[route+node]
            elif route == "x": return XIAOBAI[route+node]
            elif route == "j": return JIEJIE[route+node]
        else:
            return SPECIAL[special]

    def build_story(self) -> Generator[List[str | dict], str | None, StopIteration]:
        '''剧情文本生成器，接受Manage调控'''
        while True:
            script = self.script(self.current_route,self.current_node,self.special)
            ans = yield script
            self.current_question = script[len(script)-1]

            #接收来自Manage的信号
            follow,text = self.manager.check_state(self.current_question,ans)
            #print(follow,text)

            #进入角色线
            if follow == "entry_route":
                self.current_route = text
                self.current_node += 1
                self.current_character = self.characters[text]

            #进入特殊节点
            elif follow == "entry_special":
                self.special = text
                continue

            #进入结局
            elif follow == "entry_end":
                self.special = None
                self.current_node = text
                script=self.script(self.current_route,self.current_node,self.special)
                ans = yield script
                return StopIteration

            #玩家输入错误，重新尝试
            elif follow == "try_again":
                continue

            #进入下一个节点
            elif follow == "next_node":
                self.current_node += 1

            self.special = None