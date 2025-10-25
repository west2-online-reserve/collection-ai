import random
import time
from typing import Type
from models import get_all_pokemons, Pokemon

POKEMON_COUNT_REQUIRED = 3

if __name__ == "__main__":
    registered_pokemons: list[Type[Pokemon]] = get_all_pokemons()
    player_pokemons: list[Pokemon] = []

    while len(player_pokemons) != POKEMON_COUNT_REQUIRED:
        time.sleep(0.5)
        print(f"请选择你的{len(player_pokemons) + 1}号宝可梦：")
        for index, pokemon in enumerate(registered_pokemons):
            print(f"{index + 1}.{pokemon(bot=False).name}")
        try:
            pokemon_index = int(input()) - 1
            if pokemon_index >= len(registered_pokemons):
                raise ValueError
            selected: Pokemon = registered_pokemons[pokemon_index](bot=False)
            if selected.name in [pk.name for pk in player_pokemons]:
                print("你已经选择过这个宝可梦了!")
                continue
            else:
                print(
                    f"你选择了[{selected.name}]作为{len(player_pokemons) + 1}号宝可梦！"
                )
                player_pokemons.append(selected)

        except ValueError:
            print("输入有误，请重新输入！")

    bot_pokemons: list[Pokemon] = [
        pk(bot=True)
        for pk in random.sample(registered_pokemons, POKEMON_COUNT_REQUIRED)
    ]

    print("你选择了: " + ", ".join([pk.name for pk in player_pokemons]))
    print("电脑选择了: " + ", ".join([pk.name for pk in bot_pokemons]))

    player_pokemon: Pokemon = player_pokemons.pop(0)
    bot_pokemon: Pokemon = bot_pokemons.pop(0)
    while True:

        if player_pokemon.dead and not player_pokemons:
            print("-------------------------------------------------")
            print("你的宝可梦都寄了，你被邪恶人机击败了!")
            break

        if bot_pokemon.dead and not bot_pokemons:
            print("-------------------------------------------------")
            print("你击败了邪恶人机!")
            break

        print("-------------------------------------------------")
        time.sleep(2.0)
        if player_pokemon.dead:
            player_pokemon = player_pokemons.pop(0)
            print(f"你派出了[{player_pokemon.name}]继续战斗!")

        if bot_pokemon.dead:
            bot_pokemon = bot_pokemons.pop(0)
            print(f"电脑派出了[{bot_pokemon.name}]继续战斗!")

        print(f"[{player_pokemon.name}] VS [{bot_pokemon.name}]")
        bot_pokemon.buffs.clear()
        player_pokemon.buffs.clear()
        while True:
            player_pokemon.set_enemy(bot_pokemon)
            bot_pokemon.set_enemy(player_pokemon)
            time.sleep(1.5)

            print("--------------你的回合------------------")

            print(
                f"*{player_pokemon.name} HP: {player_pokemon.health_point}/{player_pokemon.max_health_point}"
            )

            print(
                f"*{bot_pokemon.name} HP: {bot_pokemon.health_point}/{bot_pokemon.max_health_point}"
            )
            player_pokemon.start_turn()
            if player_pokemon.check_dead():
                break
            if bot_pokemon.check_dead():
                break

            time.sleep(1.5)
            print("-------------电脑的回合-----------------")
            bot_pokemon.start_turn()
            if player_pokemon.check_dead():
                break
            if bot_pokemon.check_dead():
                break
