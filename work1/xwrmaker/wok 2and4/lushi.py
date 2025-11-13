import random
import time


class Card:
    def __init__(self, name, cost, card_type, attack=0, health=0, effect=None):
        self.name = name
        self.cost = cost
        self.card_type = card_type  # "minion", "spell", "weapon"
        self.attack = attack
        self.health = health
        self.effect = effect
        self.can_attack = False  # 随从召唤的回合不能攻击

    def __str__(self):
        if self.card_type == "minion":
            return f"{self.name} ({self.cost}费) {self.attack}/{self.health}"
        elif self.card_type == "spell":
            return f"{self.name} ({self.cost}费) 法术"
        else:
            return f"{self.name} ({self.cost}费) 武器 {self.attack}/{self.health}"


class Player:
    def __init__(self, name):
        self.name = name
        self.health = 30
        self.max_mana = 0
        self.current_mana = 0
        self.hand = []
        self.deck = []
        self.board = []  # 场上的随从
        self.weapon = None
        self.hero_power_used = False

    def draw_card(self):
        if self.deck:
            card = self.deck.pop(0)
            if len(self.hand) < 10:
                self.hand.append(card)
                print(f"{self.name} 抽到了: {card}")
                return card
            else:
                print(f"{self.name} 的手牌已满，{card.name} 被爆掉了!")
                return None
        else:
            print(f"{self.name} 的牌库空了，受到疲劳伤害!")
            self.health -= 1
            return None

    def play_card(self, card_index, target=None):
        if card_index < 0 or card_index >= len(self.hand):
            print("无效的卡牌选择!")
            return False

        card = self.hand[card_index]

        # 检查法力值
        if card.cost > self.current_mana:
            print("法力值不足!")
            return False

        # 检查场上随从数量
        if card.card_type == "minion" and len(self.board) >= 7:
            print("场上随从已满!")
            return False

        # 消耗法力值
        self.current_mana -= card.cost

        # 执行卡牌效果
        if card.card_type == "minion":
            self.board.append(card)
            print(f"{self.name} 打出了 {card.name}")

        elif card.card_type == "spell":
            print(f"{self.name} 施放了 {card.name}")
            # 这里可以添加法术效果

        # 从手牌移除
        self.hand.pop(card_index)
        return True

    def attack(self, attacker_index, target_player, target_index=None):
        # 攻击英雄
        if target_index is None:
            if self.board[attacker_index].can_attack:
                target_player.health -= self.board[attacker_index].attack
                print(
                    f"{self.name} 的 {self.board[attacker_index].name} 攻击了 {target_player.name}，造成 {self.board[attacker_index].attack} 点伤害")
                self.board[attacker_index].can_attack = False
                return True
            else:
                print("这个随从本回合已经攻击过或刚被召唤!")
                return False

        # 攻击随从
        else:
            if self.board[attacker_index].can_attack:
                attacker = self.board[attacker_index]
                defender = target_player.board[target_index]

                # 互相造成伤害
                defender.health -= attacker.attack
                attacker.health -= defender.attack

                print(f"{self.name} 的 {attacker.name} 攻击了 {target_player.name} 的 {defender.name}")

                # 检查随从是否死亡
                if defender.health <= 0:
                    print(f"{defender.name} 死亡!")
                    target_player.board.pop(target_index)

                if attacker.health <= 0:
                    print(f"{attacker.name} 死亡!")
                    self.board.pop(attacker_index)
                else:
                    attacker.can_attack = False

                return True
            else:
                print("这个随从本回合已经攻击过或刚被召唤!")
                return False


class Game:
    def __init__(self, player1_name, player2_name):
        self.players = [Player(player1_name), Player(player2_name)]
        self.current_player_index = 0
        self.turn_count = 0

        # 初始化卡组
        self.initialize_decks()

    def initialize_decks(self):
        # 创建一些基本卡牌
        basic_cards = [
            Card("闪金镇步兵", 1, "minion", 1, 2),
            Card("淡水鳄", 2, "minion", 2, 3),
            Card("破碎残阳祭司", 3, "minion", 3, 2),
            Card("冰风雪人", 4, "minion", 4, 5),
            Card("石拳食人魔", 6, "minion", 6, 7),
            Card("火球术", 4, "spell"),
            Card("奥术智慧", 3, "spell"),
            Card("神圣新星", 5, "spell"),
            Card("刺杀", 5, "spell"),
            Card("炎爆术", 10, "spell")
        ]

        # 每个玩家获得30张随机卡牌
        for player in self.players:
            player.deck = [random.choice(basic_cards) for _ in range(30)]
            random.shuffle(player.deck)

            # 起始手牌
            for _ in range(3):
                player.draw_card()

    def start_turn(self):
        self.turn_count += 1
        current_player = self.players[self.current_player_index]

        # 增加法力水晶
        if current_player.max_mana < 10:
            current_player.max_mana += 1
        current_player.current_mana = current_player.max_mana

        # 重置英雄技能和随从攻击状态
        current_player.hero_power_used = False
        for minion in current_player.board:
            minion.can_attack = True

        # 抽一张牌
        current_player.draw_card()

        print(f"\n===== 第 {self.turn_count} 回合 - {current_player.name} 的回合 =====")
        print(f"法力水晶: {current_player.current_mana}/{current_player.max_mana}")
        print(f"生命值: {current_player.health}")

    def display_game_state(self):
        current_player = self.players[self.current_player_index]
        opponent = self.players[1 - self.current_player_index]

        print(f"\n{opponent.name} ({opponent.health}生命值)")
        print("场上随从:")
        for i, minion in enumerate(opponent.board):
            print(f"  {i + 1}. {minion}")

        print(f"\n{current_player.name} ({current_player.health}生命值)")
        print("场上随从:")
        for i, minion in enumerate(current_player.board):
            attack_indicator = "可攻击" if minion.can_attack else "已攻击"
            print(f"  {i + 1}. {minion} [{attack_indicator}]")

        print("\n你的手牌:")
        for i, card in enumerate(current_player.hand):
            print(f"  {i + 1}. {card}")

    def player_turn(self):
        current_player = self.players[self.current_player_index]

        while True:
            self.display_game_state()

            print("\n你可以:")
            print("1. 打出卡牌")
            print("2. 攻击")
            print("3. 使用英雄技能")
            print("4. 结束回合")

            choice = input("请选择操作 (1-4): ")

            if choice == "1":
                # 打出卡牌
                if not current_player.hand:
                    print("你的手牌是空的!")
                    continue

                card_index = int(input("选择要打出的卡牌 (1-{}): ".format(len(current_player.hand)))) - 1

                # 如果是随从卡，直接打出
                if current_player.hand[card_index].card_type == "minion":
                    if current_player.play_card(card_index):
                        # 新召唤的随从本回合不能攻击
                        if current_player.board:
                            current_player.board[-1].can_attack = False
                else:
                    current_player.play_card(card_index)

            elif choice == "2":
                # 攻击
                if not current_player.board:
                    print("你没有随从可以攻击!")
                    continue

                attacker_index = int(input("选择攻击的随从 (1-{}): ".format(len(current_player.board)))) - 1

                # 选择攻击目标
                print("攻击目标:")
                print("1. 敌方英雄")
                opponent = self.players[1 - self.current_player_index]
                for i, minion in enumerate(opponent.board):
                    print(f"  {i + 2}. 敌方 {minion.name}")

                target_choice = int(input("选择目标 (1-{}): ".format(len(opponent.board) + 1)))

                if target_choice == 1:
                    current_player.attack(attacker_index, opponent)
                else:
                    target_index = target_choice - 2
                    current_player.attack(attacker_index, opponent, target_index)

            elif choice == "3":
                # 使用英雄技能
                if not current_player.hero_power_used and current_player.current_mana >= 2:
                    current_player.current_mana -= 2
                    current_player.hero_power_used = True
                    print("你使用了英雄技能!")
                else:
                    print("无法使用英雄技能!")

            elif choice == "4":
                # 结束回合
                break

            else:
                print("无效的选择!")

            # 检查游戏是否结束
            if self.check_game_over():
                return

    def check_game_over(self):
        for player in self.players:
            if player.health <= 0:
                winner = self.players[1 - self.players.index(player)]
                print(f"\n游戏结束! {winner.name} 获胜!")
                return True
        return False

    def play_game(self):
        print("===== 文字版炉石传说 =====")
        print("游戏开始!")

        while True:
            self.start_turn()

            self.player_turn()

            if self.check_game_over():
                break

            # 切换到下一个玩家
            self.current_player_index = 1 - self.current_player_index

        print("感谢游戏!")


# 运行游戏
if __name__ == "__main__":
    game = Game("玩家1", "玩家2")
    game.play_game()