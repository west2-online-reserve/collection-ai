'''这个文档是交互系统'''
import Pokemon,BattleSystem,random,os

class Interfact(object):
    def __init__(self) -> None:
        pass

    def opition(self):
        print('请选择你要执行的操作:\n1.开始游戏\n2.游戏资料\n3.退出游戏')
        choice = input_exam(1)
        self.clear_screen()
        if choice == '1':
            self.form_team()
            b = BattleSystem.Battle(Pokemon.player_team,Pokemon.computer_team)
            self.play_again()
        if choice == '2':
            print('请选择想查看的内容:\n1.游戏规则\n2.宝可梦大全')
            num = input_exam(2)
            if num == '1':
               self.display_rules(rules_content)
            if num == '2':
               self.display_pokemon()
        if choice == '3':
           return
    
    def form_team(self):
       print('请选择3个宝可梦用于组成你的队伍:')
       Pokemon.all_pokemons.display()
       #team = input('\n输入数字选择你的宝可梦:').split()
       team = input_exam1()
       self.clear_screen()
       for i in team:
           Pokemon.player_team.append(Pokemon.all_pokemons.pokemons[int(i)-1]) 
       print('你的宝可梦队伍如下:')
       Pokemon.player_team.display()

       cpu_team = random.sample(Pokemon.all_pokemons.pokemons,3)
       for i in cpu_team:
           Pokemon.computer_team.append(i)
       print('\n\n电脑的宝可梦队伍如下:')
       Pokemon.computer_team.display()
       print('\n-------------------------')
       

    def display_rules(self, rules_content, lines_per_page=25):  
      # 显示规则的函数  
      rules = rules_content.splitlines(keepends=True)  # 按行分割规则内容，包括换行符  
      prev = 0  
      total_lines = len(rules)  # 规则的总行数  

      while prev < total_lines:  
         current = prev + lines_per_page  
         print(''.join(rules[prev:current]))  # 显示当前页内容  
         prev = current  # 更新prev为current  
         if prev >= total_lines:  # 检查是否已经显示完所有内容  
               break  
         # 用户输入回车继续  
         input("按回车继续...")  # 等待用户按回车键  
         self.clear_screen()  # 清屏以准备显示下一页  

      print("以上为全部规则")  # 显示全部规则的提示
      input("按回车继续...")  # 等待用户按回车键
      self.clear_screen()
      self.opition() 

    def display_pokemon(self):
        while True:
            Pokemon.all_pokemons.introduction()
            flag = input("按回车继续或输入q返回:")  # 等待用户按回车键
            if flag == 'q':
                break
            self.clear_screen()  # 清屏以准备显示下一页  
        self.clear_screen()
        self.opition()


    def play_again(self):
        str = input('再来一局按回车,退出输入q:')
        if str =='q':
            return 
        Pokemon.player_team.clear()
        Pokemon.computer_team.clear()
        self.form_team()
        b = BattleSystem.Battle(Pokemon.player_team,Pokemon.computer_team)
        self.play_again()

    def clear_screen(self):  
        # 清屏函数，根据操作系统类型选择适当的清屏命令  
        os.system('cls' if os.name == 'nt' else 'clear')  # Windows使用'cls'，其他系统使用'clear' 

def input_exam(way):
    if way == 1:
        while True:
            num = input()
            if len(num)==1 and num in '123':
                break
            else:
                print('请输入有效数字:')
    elif way ==2:
        while True:
            num = input()
            if len(num)==1 and num in '12':
                break
            else:
                print('请输入有效数字:')
    return num

def input_exam1():
    print('\n输入数字选择你的宝可梦:')
    num = len(Pokemon.all_pokemons)
    while True:
        flag = 0
        team = input().split()
        for i in team:
            if len(i)!=1 or not i.isdigit() or int(i)<=0 or int(i)>num:
                print('请输入有效数字:')
                flag = 1
                break

        if len(team)<3 or len(team)>3 :
            print('请选择正确个数的宝可梦:')
            flag = 1

        if flag == 0:
            break

    return team
        
    
  
rules_content = ('''===============================  
          宝可梦对战游戏规则  
===============================  

游戏概述：  
在这个命令行宝可梦对战游戏中，玩家和电脑将各自组成一个包含3个宝可梦的队伍。玩家可以选择要出战的宝可梦进行对战。
战斗过程中，使用不同宝可梦的技能进行攻击，考验玩家的的策略与运气。  

宝可梦属性：  
1. HP（血量）：当前宝可梦的生命值。当HP降为0或以下时，宝可梦昏厥，无法继续战斗。  
2. 属性：宝可梦的属性影响战斗中的伤害和抗性。属性分为草、火、水、电等，每种属性都有其克制和被克制的关系。  
3. 攻击力：宝可梦对敌方造成的伤害值。  
4. 防御力：敌方宝可梦对其造成的伤害会减去防御力。  
5. 闪避率：宝可梦在战斗中成功躲避敌人攻击的概率。  

属性克制关系与属性被动：  
- 草属性克制水，被火克制；  被动：每回合回复 10% 最大HP值
- 火属性克制草，被水克制；  被动：每回合叠加5%攻击力
- 水属性克制火，被电克制；  被动：到伤害时，有50%的几率减免30%的伤害
- 电属性克制水，被草克制；  被动：受当成功躲闪时，可以立即使用一次技能
- 无实体属性无克制关系；    被动：无法受到负面效果

每当被克制的宝可梦受到来自克制的宝可梦的攻击时，伤害翻倍；被克制的宝可梦对克制的宝可梦造成的伤害减半。  

游戏规则：  

1. 队伍组成：  
   - 玩家和电脑各自选择3名宝可梦。  
   - 电脑随机选择宝可梦，并无法更改。  
   - 每个宝可梦具有不同的属性、攻击力、防御力、闪避率和招式。  

2. 选择出战宝可梦：  
   - 游戏开始时，玩家需要从已选择的3个宝可梦中选出1个用于出战。  

3. 回合制战斗：  
   - 游戏采用回合制，每个回合双方轮流进行攻击。  
   - 回合中可以使用的技能。  

4. 攻击及效果处理：  
   - 当出招时，计算由于属性克制关系所造成的伤害变化。  
   - 使用公式计算伤害：  
     - 伤害 = 造成的伤害 - 防御力  
     - 如果攻击方的属性克制防御方，则伤害翻倍。  
     - 如果防御方的属性被攻击方克制，则伤害减半。  

   - 属性被动：  
     - 在每回合结束时，检查宝可梦是否具备属性被动，并应用相关效果（如回血、增益等）。  

5. 闪避与命中：  
   - 每次攻击时，先判定能否命中，依据闪避率计算命中概率：  
     - 命中概率 = 100% - 对手的闪避率  
   - 如果成功躲闪，攻击无效，并根据相应的属性被动触发额外效果。  

6. 生命值管理：  
   - 每次攻击后，更新宝可梦的HP(对数据进行四舍五入)。  
   - 如果宝可梦的HP减少至0，那么该宝可梦昏厥，不能继续出战。  
   - 玩家可以选择新的宝可梦出战，直到所有宝可梦都昏厥。  

7. 游戏胜负判定：  
   - 当一方所有宝可梦均昏厥时，另一方获胜。  
   - 游戏结束后的胜负结果将显示，并询问玩家是否再来一局。  

===============================  
         祝你玩得愉快！  
===============================''')

l = Interfact()
l.opition()