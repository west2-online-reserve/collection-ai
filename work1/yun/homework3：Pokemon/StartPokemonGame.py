import pokemon

if __name__ == "__main__":
    pokemon.chooseyourpokemon()
    x = pokemon.startgame()
    if x.is_man:
        print(f"获胜者是你的 {x.name}!")
    else :
        print(f"获胜者是电脑的 {x.name}!")
    print("游戏结束！输入任意键退出...")
    input()
    