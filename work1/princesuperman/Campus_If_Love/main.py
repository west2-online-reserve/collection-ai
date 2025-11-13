from game import Game
def main():
    try:
        game=Game()
        game.show_menu()
    except KeyboardInterrupt:
        print("游戏结束")
    except Exception as e:
        print(f"有错误：{e}")        
if __name__=="__main__":
    main()