# Campus IF Love 遇到的问题与解决方法

## 1. 礼物数据缺少某个角色时取不到值

**问题**：`GIFT_EFFECTS` 中「奇怪的石头」只写了 `default`，如果直接写 `GIFT_EFFECTS[gift][self.name]`，会抛出 `KeyError`。

**解决**：用两层 `get` 做回退，先取角色专属数值，取不到时取 `default`，最后回退到 0：

```python
value = effects.get(self.name, effects.get("default", 0))
```

## 2. 输入礼物名称太麻烦

**问题**：最初要求玩家输入完整礼物名，比如「可爱的玩偶」，输入慢而且容易打错字。

**解决**：在 `_gift_menu()` 中把礼物字典的键转成列表并编号，玩家只需要输入数字；同时用 `isdigit()` 和范围判断处理非法输入
