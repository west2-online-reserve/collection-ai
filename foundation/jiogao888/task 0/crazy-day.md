# crazy-day.md
## 关于 task 1 宝可梦的碎碎念
### 1. 没有实装的回避代码
写到一半发现没有实装回避，经过头脑风暴后，写了 2 个 skill 的子类后解决。
### 2. 关于示例写了但没有实装的状态代码
写小火龙的时候发先没有实装。
### 3. 关于神秘的攻击 skill
skill 竟然是预先输入 attack 数值，成功让攻击力提升成了滚木。
### 4. 关于 debuff 伤害
debuff 伤害竟然调用的带防御力的扣血代码。于是加新的代码。
其是可以给代码加几个参数，但懒狗了。
看看[代码](https://www.bilibili.com/video/BV1GJ411x7h7/?spm_id_from=333.337.search-card.all.click&vd_source=fa2137d671ca93e2ad525f00dd3607a7)
```
def hp_lost(self,damage) -> int:
        if not isinstance(damage, int):
            damage = int(damage)

        self.hp -= damage
        printf(
            f"{self.name} received {damage} damage! Remaining HP: {self.hp}/{self.max_hp}"
        )
        if self.hp <= 0:
            self.alive = False
            print(f"{self.name} has fainted!")
        return damage
```
### 5. 敲字比敲代码累
|  | 宝可梦 | 原因 | 
|------|------|------|
| T 0 | 小火龙和杰尼龟 | 传说 300% 倍率和 20 防御力 |
| T 1 | 妙蛙种子 | 超级回血 |
|T 114514 | 皮卡丘 | 综合 110% 的电光一闪也是区完了 |

![我吗](https://ts1.tc.mm.bing.net/th/id/OIP-C.H2qPQv7RGQAsSWz2rN9l8QHaHa?w=193&h=193&c=8&rs=1&qlt=90&o=6&dpr=1.5&pid=ImgAns&rm=2)
## 结语
不想写了

