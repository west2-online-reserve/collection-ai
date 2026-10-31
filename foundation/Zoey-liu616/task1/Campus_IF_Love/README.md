# 校园·if 恋（Campus IF Love）

## 项目介绍

这是一个命令行恋爱模拟游戏。玩家会依次遇到学姐、小白和姐姐；选择进入某条路线后，可以通过对话或送礼物提升好感度，好感度达到 100 时解锁结局。

剧情与礼物数值以 [Campus_IF_Love_Story.md](Campus_IF_Love_Story.md) 为准。

## 游戏玩法

1. 在三个相遇场景中输入 `1` 进入当前角色路线，输入 `2` 则继续下一场景。
2. 进入路线后，可选择聊天、送礼、查看好感度或退出。
3. 聊天时选择 A 增加 5 点好感度，选择 B 减少 5 点好感度。
4. 礼物根据角色产生不同的好感度变化。
5. 好感度达到 100 时进入结局；好感度低于 0 时，可重新开始当前路线或切换到下一位角色。
6. 三个相遇场景都未选择时，游戏结束。

## 文件结构

```text
Campus_IF_Love/
├── main.py                  # 主入口
├── game.py                  # 场景、玩家操作和主循环
├── character.py             # 角色、对话与礼物数据
├── campus_if_love.py        # 兼容原始文件名的入口
├── Campus_IF_Love_Story.md  # 官方剧情与数值说明
├── problems.md              # 作业中遇到的问题
└── README.md
```

## 运行方法

需要 Python 3.9+。在本目录执行：

```bash
python main.py
```

也可以运行兼容入口：

```bash
python campus_if_love.py
```
