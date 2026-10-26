from .config import *
from .game import Game, confirm
import sys
import time


class Main_menu:
    def launch(self) -> None:
        for line in Pokemon_Battle:
            print(line)
            time.sleep(0.2)
        print()    
        for option in main_menu_options:
            print(option)

        while True:
            menu_choice = input(menu_choice_text)
            if menu_choice == "1":
                self.start()
            elif menu_choice == "2":
                self.quit_game()
            else:
                print(wrong_input)

    def start(self) -> None:
        game = Game()
        game.pick()

    def continue_game(self) -> None:
        pass

    def statics(self) -> None:
        pass

    def quit_game(self) -> None:
        if confirm(quit_text):
            sys.exit()
