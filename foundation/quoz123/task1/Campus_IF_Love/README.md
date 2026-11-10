# 校园·if 恋（Campus IF Love）

## 项目介绍

一个命令行恋爱模拟游戏。玩家依次遇到学姐、小白和姐姐，选择进入某条路线后，可以通过聊天或送礼物提升好感度，好感度达到 100 时解锁结局。

剧情与礼物数值以 [Campus_IF_Love_Story.md](Campus_IF_Love_Story.md) 为准。

## 游戏玩法

1. 三个相遇场景会依次出现：输入 `1` 进入当前角色路线，输入 `2` 前往下一个场景。
2. 进入路线后，可以选择聊天、送礼、查看好感度或退出游戏。
3. 聊天时根据剧本选择 A 或 B：选 A 增加 5 点好感度，选 B 减少 5 点。
4. 送礼时输入礼物编号，不同角色对同一件礼物的反应不同。
5. 好感度达到 100 时进入结局。
6. 好感度低于 0 时攻略失败，可以选择重新开始当前路线或前往下一位角色。
7. 三个场景都没有进入时，游戏结束。

## 文件结构

```text
Campus_IF_Love/
├── main.py                  # 主入口
├── campus_if_love.py        # 游戏入口（导出 Character、Game）
├── game.py                  # 场景流程、玩家操作和主循环
├── character.py             # 角色类、对话库和礼物数据
├── Campus_IF_Love_Story.md  # 官方剧本与数值说明
├── framework/
│   ├── Campus_IF_Love.py    # 官方原始框架（未修改，仅作对照）
│   └── README.md            # 框架拆分说明
├── problems.md              # 作业中遇到的问题与解决方法
└── README.md
```

## 运行方法

需要 Python 3.10+。在本目录执行：

```bash
python main.py
```

也可以运行：

```bash
python campus_if_love.py
```

## 实现说明

- 代码基于官方 `Campus_IF_Love.py` 框架拆分而来。
- `Character.talk()` 按顺序循环使用对话库，每完成一次对话推进一条。
- `Character.give_gift()` 先查角色的专属好感度，再回退到 `default`，最后回退到 0。
- `Game.story_loop()` 返回 `bool`，让上层场景可以判断是进入结局还是换到下一位角色。
- 全部代码使用 Type Hint，并通过 Pyright 的 standard 模式检查。
