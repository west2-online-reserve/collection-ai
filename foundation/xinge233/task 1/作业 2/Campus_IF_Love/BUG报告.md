# Campus_IF_Love 代码排查报告

## 一、排查方式

| 项目 | 内容 |
| --- | --- |
| 解释器 | Python 3.14.7（`C:\Users\lecoo\AppData\Local\Python\pythoncore-3.14-64\python.exe`，用 `py` 启动；`python` 指向 Microsoft Store 占位程序，本身不执行代码） |
| 被测程序 | 项目根目录 `start.py`（`python`/`py start.py` 交互式运行） |
| 方法 1 | 通读 `start.py`、`data\game.py`、`data\character.py`、`data\config.py` |
| 第一轮（改动前） | 写临时脚本给 `start.py` 自动喂 stdin，跑 23 个场景（正常流程 + 边界 + 构造存档 + 极端输入），逐个校验退出码 / Traceback / 关键输出；23/23 按预期复现（其中 8 个场景崩溃） |
| 第二轮（你改完之后，本次新增） | **重跑回归 19 个场景**（覆盖 A1~A7、B1~B7、正常流程、`saves\` 不存在、emoji 重定向输出），逐场景记录完整输入/输出；19/19 均按预期复现（4 个场景仍崩溃），结论见 §二 |
| 环境说明 | 测试会真实读写 `saves\save.json`、`saves\Qsave.json`；临时脚本与日志在排查结束后已删除，本文保留了全部复现输入序列，可照做重现 |

---

## 二、本次修改（4 处）与复测结论

### 2.1 你改了哪些地方（`git diff`：5 增 15 删）

| 文件:行 | 改动 |
| --- | --- |
| `start.py:37` | `except FileNotFoundError:` → `except Exception:`（主菜单 2「继续游戏」） |
| `start.py:48` | 同上（主菜单 3「读档」） |
| `data/game.py:203` | 同上（角色线菜单 5「读档」） |
| `data/game.py:30` | `save()` 开头新增 `BASE.mkdir(parents=True, exist_ok=True)` |
| `data/config.py:239` | `load_fail` 文案改为"读取存档失败，存档文件不存在或已损坏。" |

### 2.2 一句话结论

**这次改动只能算"止血"，不能算"修好"**：3 处放宽 `except` 让 2 个原本必崩的入口变成了"不崩溃"，但同时引入了两个新问题（**错误提示张冠李戴**、**失败路径污染内存**）；另外 2 个 P0（A3/A4）只是把崩溃换了个位置、或换成"被强行结局"；A5/A6 因为异常不在任何 `try` 覆盖范围内，仍然崩。唯一干净的收益是 `save()` 里的 `mkdir`。

### 2.3 19 个场景实测对照

| 场景 | 对应问题 | 改动前 | 改动后（本轮实测） |
| --- | --- | --- | --- |
| S01 正常流程 | — | ✅ 正常 | ✅ 正常（退出码 0） |
| S02 `save.json` 空文件 → 读档 | A1 | ❌ `JSONDecodeError` 崩溃 | ⚠️ 不崩，但打印的是**旧的** `file_error` 文案（你新改的 `load_fail` 一次都没出现） |
| S03 `save.json` 写成非 JSON → 读档 | A1 | ❌ 崩溃 | ⚠️ 同 S02 |
| S04 存档缺 `talk_times` → 读档 → 再开新游戏 | A2 | ❌ `KeyError: 'talk_times'` 崩溃 | ⚠️ 不崩，但**内存已被写坏**：`affinity` 已从存档里读成 999，异常被吞后回主菜单；接着选"开始游戏 → 学姐线"，一次对话都没做就显示 `当前与学姐的好感度为:999` + `恭喜！你和 学姐 的故事进入了结局线！` |
| S05 `talk_times="abc"` → 主菜单读档 → 聊天 | A3 | ❌ `TypeError` 崩溃 | ⚠️ 不崩，但**误报**：先打印"已从…读取存档。"（读档其实成功了），紧接着打印"读取存档失败，未找到存档文件。"并静默把玩家踢回主菜单 |
| S06 同一坏档 → 在**角色线菜单 5** 读档 → 聊天 | A3 | ❌ `TypeError` 崩溃 | ❌ **照样崩**（`character.py:22` 的 `self.talk_times >= 6`），因为 `game.py` 的 `try` 只包住赋值、没包住 `talk()` |
| S07 `Qsave.json` 的 `wife="None"` → 继续游戏 → 聊天 | A4 | ❌ `KeyError: 'None'` 崩溃 | ⚠️ 不崩，但同样误报"读取存档失败"并踢回主菜单（玩家不知道自己刚进了"None 角色线"） |
| S08 `save.json` 的 `wife="None"` → 角色线内读档 | A4 | ❌ 崩溃 | ⚠️ 变成**直接掉进结局菜单**（`游戏结束。`），玩家被强行"结局"，只能选重新开始/退出 |
| S09 输入流结束（Ctrl+Z / 无输入管道） | A5 | ❌ `EOFError` 崩溃 | ❌ **照样崩**（异常不在任何 `try` 内） |
| S10 `confirm()` 连续非法输入 | A6 | ❌ `RecursionError` 崩溃 | ❌ **照样崩**（同上） |
| S11 `talk_times=0` 存档 → 聊天 | A7 | ⚠️ 静默错乱 | ⚠️ 不变：`你正在和学姐进行第0次对话...` + 第 5 条对话内容 |
| S12 菜单 7 重开 → 主菜单「继续游戏」 | B1/B2 | ⚠️ 读回旧进度 | ⚠️ 不变：仍读回 **40**（重开前那一轮的 Qsave） |
| S13 结局菜单选 1 重开 | B3 | ⚠️ 结局菜单再出现 | ⚠️ 不变：`游戏结束。` + 选项菜单在本次会话里出现 **2 次** |
| S14 结局菜单选 3 | B4 | ⚠️ 无确认直接退出 | ⚠️ 不变：直接 `你选择离开，游戏结束。` |
| S15 连送 5 杯奶茶 | B5 | ⚠️ 0 次对话通关 | ⚠️ 不变（实测 `请用1/2作出你的回答` 出现 0 次） |
| S16 连送 3 块石头 | B6 | ⚠️ 好感度 -30 | ⚠️ 不变 |
| S17 清档确认选 2（取消） | B7 | ⚠️ 无任何反馈 | ⚠️ 不变（且仍打印只有四个字的 `快速存档`） |
| S18 输出重定向（`py start.py > log.txt`）后第 6 次对话 | **新增 C11** | （第一轮未测） | ❌ `character.py:27` 的 `print(suki_talk)` → `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f497'` |
| S19 删掉整个 `saves\` 目录后开始游戏并聊天 | **本次修复** | ❌ 首次写 `Qsave` 时 `FileNotFoundError`（静态推断：`open()` 的父目录不存在） | ✅ 正常，结束时 `saves\` 里已生成 `Qsave.json` |

### 2.4 三个必须知道的细节

1. **你改的文案没生效**：`data/config.py:239` 的 `load_fail` 从来没有被 `print` 过 —— `start.py:38`、`start.py:49`、`data/game.py:204` 打印的是 `data/config.py:244` 的 `file_error`（旧文案"未找到存档文件"）。S02/S03 断言了"新文案出现 0 次"来锁定这一点（另见 C5）。
2. **`except Exception` 包得太宽**：`start.py:30-38` 把整个 `game.story_loop()` 包进了 `try`，于是"读档成功之后游戏里发生的任何真实错误"都会被当成"读档失败"。S05/S07 是活证据：存档明明读进来了（打印了"已从…读取存档。"），下一行却报"读取存档失败"。
3. **没用裸 `except:` 是对的**：`except Exception` 不会吞掉 `SystemExit`，所以"退出游戏"仍然能正常退出（S01/S02 退出码 0 可证）；裸 `except:` 会把 `sys.exit()` 一起吞掉。但"把 `story_loop()` 包进去"这个范围问题，仍建议按 §八 的方案收窄。

---

## 三、结论摘要

| 编号 | 级别 | 一句话描述 | 稳定复现 |
| --- | --- | --- | --- |
| A1 | 高 | 存档文件损坏/为空/非 JSON → `JSONDecodeError` 直接崩溃 | ✅ |
| A2 | 高 | 存档缺字段 → `KeyError` 崩溃 | ✅ |
| A3 | 高 | 存档字段类型不对 → `TypeError` 崩溃 | ✅ |
| A4 | 高 | 存档 `wife` 为 `"None"` → 聊天时 `KeyError: 'None'` 崩溃 | ✅（需异常存档） |
| A5 | 中 | 输入流结束（Ctrl+Z / 管道）→ `EOFError` 崩溃 | ✅ |
| A6 | 中 | `confirm()` 用递归重试 → 大量非法输入触发 `RecursionError` | ✅ |
| A7 | 中 | `talk_times <= 0` 的存档 → 负数索引取到"第 5 条对话"，静默错乱 | ✅ |
| B1 | 高 | 「重新开始游戏」不清理/不刷新快速存档 → 「继续游戏」能读回旧进度 | ✅ |
| B2 | 高 | 角色线菜单选 7「重新开始」会把菜单/结局判定嵌套叠加 | ✅ |
| B3 | 中 | 结局菜单会反复出现；选 1「重新开始」后仍回到结局菜单 | ✅ |
| B4 | 中 | 菜单 3 语义冲突（查看好感度 / 退出游戏），且结局菜单退出无二次确认 | ✅ |
| B5 | 中 | 送礼物无任何限制 → 5 次奶茶直接通关，跳过全部对话 | ✅ |
| B6 | 低 | 好感度无下限（可到负数），也没有失败结局 | ✅ |
| B7 | 中 | 清档确认选「取消」后毫无反馈；`clean_Qsave` 文案只有两个字 | ✅ |
| C1~C11 | 低 | 重复定义、死代码、路径写法不一致、文本硬编码、未 `strip()`、emoji 在重定向输出时崩溃等 | 静态确认（C11 另经 S18 实测） |

> **本次改动后的状态（第二轮 19 场景实测）**：A1/A2/A3/A4 从"必崩"变成"时崩时被吞"（A1 提示张冠李戴、A2 内存被写坏、A3/A4 在角色线内读档时仍崩 / 变成"被强行结局"）；A5、A6、A7 与 B1~B7 **完全没变**；新增 C11。逐项证据见 §二.3。

> 说明：`遇到的问题.md` 里写的"残留了一些触发概率较低的 bug"——经过排查，**真正会"随机出现"的是 A1~A4（存档被写坏/被改过就必崩）和 B1/B2（重开相关状态残留）**，剩下的是可稳定触发的逻辑问题。

---

## 四、崩溃类问题（原为直接打印 Traceback 后退出）

### A1 存档不是合法 JSON → 崩溃

> **本次改动后（S02/S03 实测）**：不再崩溃了，会打印 `file_error` 后回到主菜单 —— 但打印的是**旧文案**"读取存档失败，未找到存档文件。"。你改的 `load_fail` 没有被任何 `print` 引用（见 C5），所以玩家看到的提示还是"没找到存档文件"，而不是"存档已损坏"。

- 位置：`data/game.py:47-58`（`load()` 里的 `json.load`）；调用方 `start.py:29-38`、`start.py:39-49`、`data/game.py:198-204` 只捕获了 `FileNotFoundError`。
- 现象：程序抛 `json.decoder.JSONDecodeError` 直接终止（退出码 1），看不到任何游戏提示。
- 复现步骤（快速存档版）：
  1. 新建 `saves\Qsave.json`，内容留空（或写成 `这不是合法JSON{{{`）；
  2. 运行 `py start.py` → 输入 `2`（继续游戏）→ 崩溃。
- 复现步骤（普通存档版）：把 `saves\save.json` 写成任意非 JSON → 主菜单输入 `3` → 输入 `1` → 崩溃。
- 证据：

  ```
  json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
  ```

- 为什么"概率低但会发生"：上次运行被强杀、断电、磁盘写入中断，都会留下空文件或半个 JSON。
- 建议：`load()` 里把 `json.load` 包在 `try/except (json.JSONDecodeError, OSError, ValueError)` 中，失败时返回 `None` 并提示"存档已损坏，请重新开始"，同时把损坏文件重命名备份（如 `save.json.bak`），不要让主流程崩溃。

### A2 存档缺字段 → 崩溃

> **本次改动后（S04 实测）**：不再崩溃，但变成了更隐蔽的问题 —— `current_target`、`affinity` 已经写进内存，只有 `talk_times` 赋值失败被吞掉；玩家随后选"开始游戏 → 学姐线"，在**一次对话都没做**的情况下就会看到 `当前与学姐的好感度为:999` 和 `恭喜！你和 学姐 的故事进入了结局线！`。这是"把 `except` 放宽在赋值中途"的典型后果：内存里留下改了一半的状态，而且没有任何回滚。

- 位置：`start.py:32-34`、`data/game.py:200-202` 直接 `data["wife"]` / `data["affinity"]` / `data["talk_times"]`。
- 复现：`saves\save.json` 内容为 `{"wife": "小白"}` → 主菜单 `3` → `1` → 崩溃。
- 证据：`KeyError: 'affinity'`
- 建议：统一用 `data.get(...)` + 默认值，或写一个 `_validate_save(data)` 函数，字段缺失/非法就直接判定为"存档损坏"。

### A3 存档字段类型不对 → 崩溃

> **本次改动后（S05/S06 实测）**：从**主菜单**读档这条路被 `except Exception` 吞掉，但会**误报**"读取存档失败"（其实存档已经成功读进来了，屏幕上刚打印过"已从…读取存档。"）；从**角色线菜单 5** 读档这条路不在 `try` 范围内，照样崩在 `character.py:22` 同一行。

- 位置：`data/character.py:22`（`self.talk_times >= 6`）。
- 复现：`saves\save.json` = `{"wife": "小白", "affinity": 10, "talk_times": "abc"}` → 主菜单 `3` → `1`（读档）→ 进入角色线后输入 `1`（聊天）→ 崩溃。
- 证据：

  ```
  TypeError: '>=' not supported between instances of 'str' and 'int'
  ```

- 建议：读档后强制类型转换（`int(...)`）并做范围校验，转换失败就报"存档损坏"。

### A4 存档 `wife` 为 `"None"` 时聊天 → `KeyError: 'None'`

> **本次改动后（S07/S08 实测）**：从**主菜单**"继续游戏"进入 → 异常被吞掉、误报"读取存档失败"后回主菜单（玩家不知道自己刚进了"None 角色线"）；从**角色线菜单 5** 读档进入 → 因为 `current_target` 已经变成 `characters["None"]`，`story_loop` 尾部的结局判定立刻成立，玩家会**直接掉进"游戏结束。"结局菜单**，连聊天都没机会。

- 位置：`data/character.py:31`、`23`、`27` 使用 `DIALOGUES[self.name]` / `random_talk[self.name]`，但 `data/config.py` 的 `DIALOGUES`（第 3-85 行）与 `random_talk`（第 87-91 行）**都没有 `"None"` 键**；而 `data/game.py:69/71`、`248-251` 却把 `Character("None", "无")` 作为占位角色放进 `characters`。
- 复现：
  1. `saves\Qsave.json` = `{"wife": "None", "affinity": 0, "talk_times": 1}`；
  2. 运行 → 主菜单 `2`（继续游戏）→ 角色线菜单 `1`（聊天）→ 崩溃；
  3. 用同样的 `save.json` + 主菜单 `3` `1` 也会崩。
- 证据：`KeyError: 'None'`
- 诚实说明：**我最初以为"正常玩就能触发"，实测没有触发**——因为 `story_loop` 的入口都会先经过场景函数设置真实角色，而菜单 7「重开」后外层循环会立刻被结局判定截住（见 B2）。所以 A4 属于"存档数据未被校验"这一族：只要存档被改过/来自旧版本就会 100% 崩溃。
- 建议：`talk()` 开头先判断 `if self.name not in DIALOGUES: print(...); return`，或者给 `DIALOGUES`/`random_talk` 补 `"None"` 兜底文案，更彻底的做法是给 `Character` 加 `is_placeholder` 标记，`story_loop` 一进来就拒绝占位角色。

### A5 输入流结束 → `EOFError` 崩溃

- 位置：所有 `input()`（`start.py:16`、`data/game.py:13/87/108/137/167/230`、`data/character.py:36/55`）。
- 复现：`py start.py` 后按 `Ctrl+Z` 再回车（或 `type nul | py start.py` 这种无输入管道）→ 崩溃。
- 证据：`EOFError: EOF when reading a line`
- 建议：封装一个 `ask(prompt)` 辅助函数，捕获 `EOFError`/`KeyboardInterrupt` 后优雅退出（或当作非法输入处理）。

### A6 `confirm()` 递归重试 → `RecursionError`

- 位置：`data/game.py:11-20`（第 20 行 `return confirm(action=action)`）。
- 复现：主菜单输入 `4`（退出游戏）→ 连续输入 1000 次以上非法字符（如 `x`）→ 崩溃；期间程序会把确认提示刷屏。
- 证据：`RecursionError: maximum recursion depth exceeded`
- 建议：改成 `while True:` 循环重问，不要用递归。

### A7 `talk_times <= 0` 的存档 → 静默取到错误的对话

- 位置：`data/character.py:31`（`DIALOGUES[self.name][self.talk_times - 1]`）。
- 复现：`saves\Qsave.json` = `{"wife": "学姐", "affinity": 10, "talk_times": 0}` → 主菜单 `2` → `1`（聊天）。
- 证据（实测输出）：

  ```
  你正在和学姐进行第0次对话...
  今天的社团活动结束了，你要不要和我一起收拾？   ← 因为 [0-1] == [-1]，取到了第 5 条
  ```

- 说明：Python 负数索引不报错，所以这类"存档被改坏"的情况不会崩溃，只会**悄悄给出错误内容**，比崩溃更难排查。
- 建议：读档时校验 `1 <= talk_times <= len(DIALOGUES[name])`，越界就夹紧到合法范围。

---

## 五、流程 / 状态类问题（不崩溃，但逻辑不对）

> **本次修改后的状态（第二轮 19 场景实测）**：A1/A2/A3/A4 从"必崩"变成"时崩时被吞"——A1 变成"不崩但提示张冠李戴"，A2 变成"不崩但内存被写坏"，A3/A4 在**角色线内读档**时仍然崩 / 变成"被强行结局"；A5、A6、A7 与 B1~B7 **完全没有变化**；另外新增一条 C11（emoji 在重定向输出时崩溃）。逐项证据见 §二.3。

### B1【最值得修】「重新开始游戏」不清理快速存档 → 重开后还能读回旧进度

- 位置：`data/character.py` 无涉；`data/game.py:154-162`（每轮 `story_loop` 开头写 `Qsave`）、`data/game.py:209-212`（菜单 7 重开）、`data/game.py:231-234`（结局菜单 1 重开）、`data/game.py:244-251`（`reset()` 只重置内存对象，不碰 `saves\`）。
- 现象：重开后 Qsave 里仍是重开前那一轮的数据；而且因为 Qsave 是在每轮菜单**之前**写入的，礼物加分发生在写入之后，所以 Qsave 会"落后一步"，玩家会莫名其妙回档。
- 复现步骤（照抄即可，`saves\` 先清空）：

  ```
  运行 py start.py
  输入：1            # 开始游戏
  输入：1            # 场景一：学姐线
  输入：（2 回车 3 回车）重复 5 次   # 送 5 次奶茶 → 好感度 100
  输入：1            # 结局菜单：重新开始游戏
  输入：2 2 2        # 新一局三个场景全部拒绝 → 打印"死现充别玩 galgame"
  输入：2            # 结局菜单：退出至主菜单
  输入：2            # 主菜单：继续游戏
  输入：3            # 角色线菜单：查看好感度
  ```

- 证据（实测输出）：

  ```
  学姐 的好感度变化 20 -> 当前好感度：100
  恭喜！你和 学姐 的故事进入了结局线！
  游戏结束。
  ...
  你选择重新开始游戏，游戏重置。
  ...
  啥，眼前三妹子都不要？？死现充别玩 galgame
  游戏结束。
  ...
  校园·if 恋（Campus IF Love）
  请输入选项：你要做什么？ ...
  当前与学姐的好感度为:80      ← 重开之后，居然读回了旧进度 80
  ```

- 建议：`reset()` 里同时删除/覆盖 `saves\Qsave.json`（并可选删除 `save.json`）；或者把 Qsave 的写入挪到"选项处理之后"，保证存档永远反映最新状态。

### B2 角色线菜单选 7「重新开始」→ 菜单/结局判定嵌套叠加

- 位置：`data/game.py:209-212`（`reset()` + `self.start()` 后**没有 `return`/`continue`**），紧接着 `data/game.py:219-225` 的结局判定会在同一次循环里执行。
- 现象：在角色线里点"重新开始游戏"，走完新一局的开场后，玩家会看到"游戏结束"菜单而不是干净的主菜单；在那里再选 1 又会再嵌一层。
- 复现：

  ```
  输入：1 1          # 开始游戏 → 学姐线
  输入：7            # 角色线菜单：重新开始游戏
  输入：2 2 2        # 三个场景全部拒绝
  输入：1            # 结局菜单：重新开始游戏
  输入：2 2 2        # 又是三个场景全部拒绝
  输入：3            # 结局菜单：退出游戏
  ```

- 证据（实测输出节选）：

  ```
  你选择重新开始游戏，游戏重置。
  ========== 游戏开始：校园 if·恋 ==========
  ...
  啥，眼前三妹子都不要？？死现充别玩 galgame
  游戏结束。                     ← 外层 story_loop 的结局判定被触发
  1. 重新开始游戏
  2. 退出至主菜单
  3. 退出游戏
  请输入选项：你选择重新开始游戏，游戏重置。   ← 又嵌一层
  ```

- 建议：菜单 7 处理完后 `continue`（或 `return self.start()`），让新一局跑完直接回到主菜单；结局判定的触发条件也别用"对象是否等于 `characters["None"]"` 这种隐式约定。

### B3 结局菜单会反复出现，重开也不能真正"重开"

- 位置：`data/game.py:219-225`（`self.current_target == self.characters["None"]` 或 `check_ending()` 为真就进结局菜单）与 `data/game.py:226-242`。
- 现象：好感度一旦 ≥100，`check_ending()` 每轮都会再打印一次"恭喜！你和 X 的故事进入了结局线！"；结局菜单选 1（重新开始）执行 `reset()+start()` 后回到同一个 `while True`，而 `current_target` 此时是新 `characters["None"]` → 判定为真 → **又回到结局菜单**，玩家不选 2/3 就出不去。
- 复现：接 B2 的步骤，在"游戏结束"菜单里只反复选 `1`，即可看到菜单无限循环。
- 建议：`check_ending()` 只应在"好感度刚刚跨过 100"的那一次触发（加个 `self.ending_shown` 标记）；结局菜单选 1 后应 `return`，不要留在结局循环里。

### B4 菜单 3 语义冲突：`查看好感度` 与 `退出游戏` 撞车（而且退出没有二次确认）

- 位置：`data/config.py:211-220`（角色线 `menu`：3 = 查看好感度）与 `data/config.py:227`（`ending_options`：3 = 退出游戏），两者都用 `data/config.py:209` 的同一个提示语 `loop_option = "请输入选项："`。
- 复现：

  ```
  输入：1 1                # 开始 → 学姐线
  输入：（2 回车 3 回车）× 5   # 5 次奶茶 → 好感度 100，进入结局线
  输入：3                  # 玩家以为"查看好感度"
  ```

- 证据（实测输出）：

  ```
  学姐 的好感度变化 20 -> 当前好感度：100
  恭喜！你和 学姐 的故事进入了结局线！
  游戏结束。
  1. 重新开始游戏
  2. 退出至主菜单
  3. 退出游戏
  请输入选项：你选择离开，游戏结束。      ← 直接退出游戏，没有任何"确定要退出吗"
  ```

- 影响：玩家只输入了一个数字就"游戏关闭"，会觉得程序崩了。注意角色线菜单的 6（离开）是有 `confirm` 的，这里没有，属于不一致。
- 建议：给结局菜单单独的提示语（如 `ending_option_text = "请输入选项(1-3)："`），并且退出游戏也走 `confirm()`。

### B5 送礼物没有任何限制 → 5 次奶茶直接通关

- 位置：`data/character.py:52-68`（`give_gift`），`data/game.py:174-175`（菜单 2 直接调用）。
- 现象：礼物不消耗对话次数、不设冷却/上限，也没有"同一礼物重复送递减效果"。
- 复现：开始 → 学姐线 → `2` `3`（奶茶，+20）重复 5 次 → 好感度 100 → 直接进入结局线，**一次对话都没有**。
- 证据：见上面 B4 的输出（第 5 次奶茶后立刻"恭喜！…进入了结局线！"）。
- 建议：给每个角色/每轮加"送礼物次数上限"或让重复礼物收益递减；或者在 `GIFT_EFFECTS` 里加"可用次数"。

### B6 好感度可以为负，且没有负面结局

- 复现：开始 → 学姐线 → `2` `4`（奇怪的石头，-10）连送 3 次 → `3` 查看好感度。
- 证据：`当前与学姐的好感度为:-30`
- 建议：`change_affinity` 里做 `max(0, ...)` 或 `max(-N, ...)` 夹紧，并考虑加一个"好感度过低 → 失败结局"的分支（否则负好感毫无意义）。

### B7 清档确认选「取消」后毫无反馈；`clean_Qsave` 文案不完整

- 位置：`start.py:18-28`（`if confirm(...)` 为 False 时直接回到菜单，什么都不做，也没提示）；`data/config.py:245-246`（`clean_Qsave_tips` 提示要对齐，但 `clean_Qsave = "快速存档"` 只有四个字被直接 `print` 出来）。
- 复现：玩到有 Qsave → 回主菜单 → 输入 `1` → 在"你确定要开始游戏吗？"处输入 `2`（取消）→ 屏幕直接重新打印主菜单，没有任何解释。
- 另外：这里的问题是"是否清除快速存档"，但 `start.py:21` 传的 `action=menu_text[0]` 是 `"开始游戏"`，提示语变成了"你确定要开始游戏吗？"，与实际含义不符（真正的 action 应该是 "清除快速存档"）。
- 建议：
  1. 取消时打印一句"已取消，快速存档保留"；
  2. `confirm(action="清除快速存档")`；
  3. `clean_Qsave` 改成完整句子，如 `"已清除快速存档。"`。

---

## 六、代码一致性 / 可维护性问题（不影响运行）

| 编号 | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| C1 | `data/config.py:105` 与 `:231` | `wrong_output` 定义了两次，后被前者覆盖，第 105 行是死代码（两份内容还不完全一样：第 105 行没有 `"None"` 键） | 只保留一份，删除第 105 行 |
| C2 | `data/game.py:8` vs `start.py:9` | 存档目录两种写法：`Path(__file__).resolve().parent / ".." / "saves"` 和 `Path(__file__).resolve().parent / "saves"`（两边都做了 `resolve()`，结果一致，但仍是重复定义） | 统一放到 `config.py` 或一个 `paths.py` 里定义一次 |
| C3 | `data/character.py:5-16`、`data/game.py:113/142` | `True_answer`/`False_answer`/`talk_answer`/`gift_name` 都是运行时动态创建的属性，`__init__` 里没声明；`self.phrase = 1` 是死代码（只在小白/姐姐分支赋值，从未被读取） | 在 `__init__` 里初始化，删掉 `phrase` |
| C4 | `data/character.py:58`、`data/character.py:18-19` | 硬编码文本没走 `config.py`（"你选择了不送礼物。"、"1"/"2" 答案字面量），与 README 里"所有文本集中到一个文件"的目标不一致 | 全部移到 `config.py`，答案也定义成常量 |
| C5 | `data/config.py:239` vs `:244` | `load_fail` 定义了但**从未被使用** —— `start.py`/`data/game.py` 打印的是 `file_error`。本次修改把 `load_fail` 的文案换成了"存档文件不存在或已损坏"，但程序打印的还是 `file_error` 的旧文案，**改了等于没改**（S02/S03 实测：新文案出现 0 次） | 两个变量合并成一个，并确认改的是真正被 `print` 的那个 |
| C6 | `start.py:4-7`、`data/game.py:11/23-28` | `start.py` 的 `typing`/`json` 未使用；`typing.Literal["self.current_target"]`、`Literal["退出游戏", ...]` 这种写法标注的是"字符串字面量类型"，不是真实类型，IDE 提示会误导 | 删未用 import；类型标注改成 `str`/`Literal["save","Qsave"]` 这类真正有意义的 |
| C7 | `data/game.py:11` | `confirm()` 的 `Literal` 只允许 4 个值，但调用时传入了 `menu_text[0]="开始游戏"` | 标注与实现对齐 |
| C8 | 所有 `input()` | 输入没有 `strip()`：在礼物菜单输入 `" 3"`（前面带一个空格）会被判为非法输入并重新询问 | `input(...).strip()` |
| C9 | `data/character.py:47-48` | 聊天里保留了 `answer` 作弊码（输入它会直接打印正确选项） | 这是调试残留，建议删除或改成只在 `__debug__` 下生效 |
| C10 | `data/character.py:36-50` vs `data/game.py:86-100` | 非法输入的提示不统一：场景/礼物会额外打印"请输入正确的选项！"，聊天里只打印 `wrong_output[...]`（"你想干啥呀?"），玩家不知道要输入 1/2 | 统一提示语 |
| C11 | `data/config.py:93` + `data/character.py:27` | `suki_talk = "sukisuki💗"` 里的 emoji 在非 UTF-8 输出下会崩：把 stdout 重定向到文件/管道（`py start.py > log.txt`）时 Windows 用 GBK 编码，`print(suki_talk)` 抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f497'`（S18 实测；在控制台里直接运行不受影响） | 去掉 emoji，或在入口处 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` |

---

## 七、确认"没问题"的部分（避免误判）

> 说明：下面条目里的 `Sxx` 是**第一轮** 23 场景的编号；本轮 19 场景的编号与证据见 §二.3。

以下是实测通过、行为正确的逻辑，**不需要改**：

1. 主菜单输入非法值 → 提示"请输入正确的选项！"并重新显示主菜单（`S01` 通过，退出码 0）。
2. 没有存档时选"继续游戏" / "读档" → 都只打印"读取存档失败，未找到存档文件。"然后回到菜单，没有崩溃（`S02`、`S03` 通过）。
3. 存档/读档本身的数据是对的：玩一次对话（好感 5、talk_times 2）→ 存档 → `saves\save.json` 与 `saves\Qsave.json` 内容均为 `{"wife": "学姐", "affinity": 5, "talk_times": 2}`（`S18` 通过）。
4. 场景一/二/三全部拒绝 → 正确打印"啥，眼前三妹子都不要？？死现充别玩 galgame"。
5. 连续对话 6 次后进入随机闲聊分支（`self.talk_times >= 6`）逻辑正确：第 6 次对话 +5 好感、talk_times 增至 7（`S09` 通过）。
6. 礼物选择 `9` / `" 3"` 等非法输入 → 正确提示并重新询问；选 `8` → "你选择了不送礼物。"并返回菜单（`S12`、`S12b` 通过）。
7. 存档正常时，好感度 80 / -30 / 100 等数值计算无误（`S11`、`S20`、`S10` 通过）。
8. 三个角色的 `DIALOGUES`（各 5 条）、`GIFT_EFFECTS`（7 种礼物 × 3 角色）键值齐全，没有缺失键（除 `"None"` 以外，见 A4）。

9. `saves\` 目录不存在时也能正常玩下去：`save()` 里新增的 `BASE.mkdir(parents=True, exist_ok=True)` 生效，第一次进入角色线就会自动创建目录并写出 `Qsave.json`（本轮 S19 实测通过）。

---

## 八、建议的修复优先级

### 现在最该做的：把"放宽 except"换成"`load()` 自己校验"

本次改动的思路是"让 `except` 把崩溃吞掉"，实测结果是 4 个 P0 里 2 个变成了新的错误形态（误报 / 内存污染）、2 个只是换了崩溃位置。推荐把 A 组一次改成"`load()` 负责校验、调用方负责提示"：

1. `data/game.py:47-58` 的 `load()`：读文件只捕 `FileNotFoundError`；解析只捕 `(json.JSONDecodeError, UnicodeDecodeError, OSError)`；然后**全量校验**（`wife` 必须能在 `characters` 里查到、`affinity`/`talk_times` 必须是 `int`、`talk_times >= 1`），任一不满足就 `print(load_fail)`（这时你改的那句文案才真正用上）并 `return None`。
2. `Game` 增加 `apply_save(data)`，把 `current_target / affinity / talk_times` 三处赋值收在一个函数里（先判合法再赋值，避免 A2 那种"改一半"的状态）。
3. 三个调用点（`start.py:29-38`、`start.py:39-49`、`data/game.py:195-204`）统一改成：

   ```python
   data = load(...)
   if data is None:
       continue          # 或 return，看所处循环
   game.apply_save(data)
   game.story_loop()
   ```

   并把 `game.story_loop()` **移出 `try`**（它现在被包在 `except Exception` 里，导致游戏内任何真实错误都误报成"读取存档失败"，S05/S07 就是活证）。
4. 注意**不要用裸 `except:`** —— `except Exception` 至少不会吞掉 `SystemExit`/`KeyboardInterrupt`，"退出游戏"和 Ctrl+C 还能正常工作；要缩的只是"包住整个 `story_loop()`"这个过大的范围。

### 其余优先级（与第一轮一致，本轮实测仍全部存在）

1. **P1（影响体验/存档可信度）**：B1、B2、B3、B4 —— 重开要清 Qsave；菜单 7/结局菜单 1 处理完要 `continue`/`return`；结局判定加"只触发一次"的标记；结局菜单给独立提示语并对退出加 `confirm`。
2. **P2（玩法平衡与健壮性）**：B5、B6、A5、A6、A7 —— 礼物次数限制、好感度夹紧、把 `input()` 包进一个 `ask()` 里处理 `EOFError`、`confirm` 改循环重问、`talk_times` 范围校验。
3. **P3（整洁度）**：B7 + C1~C11 —— 文案补全（先把 `file_error` / `load_fail` 合并成一个）、重复代码合并、死代码清理、类型标注修正、emoji 处理。

---

## 九、如何复测

- 单次手动验证：在项目根目录执行 `py start.py`，按上面每条「复现步骤」输入即可。
- 构造异常存档：直接编辑 `saves\save.json` / `saves\Qsave.json`（这两个文件由程序在首次存档/进入角色线时自动创建）：
  - 空文件 → 复现 A1
  - `{"wife": "小白"}` → 复现 A2
  - `{"wife": "小白", "affinity": 10, "talk_times": "abc"}` → 复现 A3
  - `{"wife": "None", "affinity": 0, "talk_times": 1}` → 复现 A4
  - `{"wife": "学姐", "affinity": 10, "talk_times": 0}` → 复现 A7
- 自动化复测：本次使用的临时脚本思路是"用 `subprocess.run([sys.executable, "start.py"], input=..., capture_output=True, encoding="utf-8")` 喂输入，然后断言 stdout 中是否出现关键文本、stderr 中是否出现 `Traceback`"。需要的话可以把它做成长期保留的 `tests/` 脚本（本次按约定已删除）。

### 第二轮（本次改动后）的自动化复测方法

- 起子进程：`subprocess.Popen([sys.executable, "-u", "start.py"], stdin=PIPE, stdout=PIPE, stderr=STDOUT)`；
- **按特征文本驱动**：先等到某个只在特定菜单里出现的文本再喂下一步输入（如 `7. 重新开始游戏` 只属于角色线菜单、`3. 退出游戏` 只属于结局菜单、`1. 确定` 只属于确认提示），这样就不用靠人肉数"菜单嵌套了几层"；
- 断言维度：是否出现 `Traceback`、关键输出片段、关键数值正则（如 `当前与学姐的好感度为[:：]\s*999`）、退出码、结束时 `saves\` 里的文件；
- 想复现 S18 需要给子进程设 `PYTHONIOENCODING=gbk`，等价于 `py start.py > log.txt` 的输出环境；
- 想手工复现本轮改动带来的新行为，按这几条输入即可（都能稳定复现）：
  1. `save.json` = `{"wife": "学姐", "affinity": 999}` → 主菜单 `3` `1`（读档，会报读取失败）→ 回主菜单后 `1` `1`（开始 → 学姐线）→ `3`（查看好感度）→ 立刻"恭喜！你和 学姐 的故事进入了结局线！"（S04，A2 的内存污染）
  2. `save.json` = `{"wife": "小白", "affinity": 10, "talk_times": "abc"}` → `1` `1`（开始 → 学姐线）→ 菜单 `5` `1`（读档）→ 菜单 `1`（聊天）→ `TypeError` 崩溃（S06）
  3. `save.json` = `{"wife": "None", "affinity": 0, "talk_times": 1}` → `1` `1` → 菜单 `5` `1` → 直接出现"游戏结束。"结局菜单（S08）
  4. 正常玩到好感度 95、第 6 次对话 → 用 `py start.py > log.txt` 运行会崩在 💗（S18）

### 补充验证：不动任何存档文件的「正常流程」（N1~N4 实测）

要回答的问题是：**正常玩、不手动改存档，还会不会出异常？** 实测结论：**正常的菜单输入确实不会崩**（N1/N2），但有 3 个「正常操作也会崩」的入口（N3/N4 与 Ctrl+C），另有 1 个「正常玩不到、逻辑上却会崩」的入口（A6 需要连续约 1000 次非法确认输入）。

| 编号 | 场景（全程不碰 `saves\` 里的文件） | 结果 |
| --- | --- | --- |
| N1 | 完整正常流程：开始游戏 → 学姐线 → 3 次奶茶（好感 60）+ 7 次对话（第 7 次走进 `suki_talk` 💗 分支）→ 好感 100 → 结局菜单 → 退出至主菜单 → 退出游戏 | **通过**：退出码 0、输出中 `Traceback` 出现 0 次、且**没有**出现"读取存档失败"误报 |
| N2 | 每个菜单都喂非法输入（`abc`、单个空格、空行、`💗`、`-1`、`0`、30 位数字、3000 个 `z`），再正常「存档 → 读档 → 取消离开 → 离开」 | **通过**：退出码 0、`Traceback` 0 次；非法输入全部被 `wrong_output` 接住，读档打印"已从…读取存档。"（`file_error` 文案 0 次） |
| N3 | 输入流结束（关闭 stdin，等价于在控制台按 `Ctrl+Z` 再回车） | **崩溃**：`start.py:16 EOFError: EOF when reading a line`（= A5） |
| N4 | 与 N1 完全相同的输入，但把输出重定向到文件（`PYTHONIOENCODING=gbk`，等价于 `py start.py > log.txt`） | **崩溃**：走到 💗 分支执行 `print(suki_talk)` 时抛 `UnicodeEncodeError`（= C11） |

- 判定方法：沿用第二轮那套「按特征文本驱动」的脚本（子进程 `Popen` + 等特征串再喂输入），断言维度为 `Traceback` 出现次数、退出码、以及 `file_error` 文案是否出现 —— 最后一条专门用来证明「把 `except` 放宽之后，**正常流程不会误报**」。
- 编码前提：N1/N2 里子进程的输出编码被显式设为 UTF-8；一旦换成 GBK（重定向到文件时的默认行为），同一条路径就会退化成 N4 的崩溃。
- `Ctrl+C`（`KeyboardInterrupt`）没法注入子进程自动测试，但可从代码上确定：全项目**没有任何一处**捕获 `KeyboardInterrupt` / `EOFError`，只有 `start.py:37`、`start.py:48`、`data/game.py:203` 三处 `except Exception`（且都是捕读档失败），所以 Ctrl+C 会按 Python 默认行为打印 traceback 退出（`main_menu` / `loop_option` / 场景提示处都一样）。
- 因此「不会崩」不等于「没毛病」：B1~B7 那些问题在 N1/N2 里全程存在（重开读回旧档、5 杯奶茶通关、好感可负、结局菜单重复等），只是它们**不抛异常**而已。
