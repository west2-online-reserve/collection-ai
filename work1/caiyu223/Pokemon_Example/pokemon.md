# pokemon程序说明

## 闪避反击  
写皮卡丘时加入的机制，pokemon类中加入self.avoid_state,用以标记是否闪避。 
加入passive_attack(self,play)用以启动闪避反击。（写的时候还在担心在play类运行中能不能传入本身，发现是可以的，只要传入self即可）

## 延迟技能
写小火龙的skills.FlameCharge()时加的机制。增加pokemon.delay_skill,改写play.player_use_skills()加了个延迟技能判断，避免延迟技能被闪避，并将技能传入pokemon.delay_skill中，在下一回合的play.player_use_skills()中直接使用pokemon.delay_skill，跳过技能选择。
    
     
在写pokemon时，这两个机制的实现花了我的最多时间，实在是太阴了。都是pokemon对play的进行造成影响。搞得要两边来回加代码，而且逻辑还很容易出问题，报错一堆。╮(╯_╰)╭
