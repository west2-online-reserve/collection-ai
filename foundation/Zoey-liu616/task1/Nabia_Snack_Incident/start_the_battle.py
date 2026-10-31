"""运行娜比娅偷吃事件的完整战斗。"""

from longmen_vs_nabiya import main_battle_loop


if __name__ == "__main__":
    print("「嗯？有塞壬的气息……就在前方！」\n")
    result = main_battle_loop(pause_seconds=0.2)
    if result == "nagato":
        print("\n「哼，知道余的厉害就好！港区的和平，由余来守护！」")
    elif result == "nabiya":
        print("\n「这次先放过你，娜比娅！」")
    else:
        print("\n战斗暂时没有分出胜负。")
