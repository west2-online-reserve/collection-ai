from data.game import *
from data.config import *
from data.character import *
import typing
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parent / "saves"

game = Game()
while True:
    for i in main_menu:
        print(i)

    main_choice = input(main_choice_text)
    if main_choice == "1":  # 开始
        if (BASE / "Qsave.json").exists():
            
            print(clean_Qsave_tips)
            if confirm(action=menu_text[0]):
                (BASE / "Qsave.json").unlink()
                print(clean_Qsave)
                game.reset()
                game.start()

        else:
            game.start()
    elif main_choice == "2":  # 继续游戏
        try:
            data = load(load_method="Qload")
            game.current_target = game.characters[data["wife"]]
            game.current_target.affinity = data["affinity"]
            game.current_target.talk_times = data["talk_times"]
            game.story_loop()

        except FileNotFoundError:
            print(file_error)
    elif main_choice == "3":  # 读档
        if confirm(menu_text[2]):
            try:
                data = load()
                game.current_target = game.characters[data["wife"]]
                game.current_target.affinity = data["affinity"]
                game.current_target.talk_times = data["talk_times"]
                game.story_loop()

            except FileNotFoundError:
                print(file_error)
    elif main_choice == "4":  # 退出游戏
        if confirm(menu_text[3]):
            print(leave_text)
            sys.exit(0)
    else:
        print(wrong_output_tip)
