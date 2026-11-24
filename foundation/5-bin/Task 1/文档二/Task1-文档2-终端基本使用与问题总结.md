# Task 1 文档 2：终端基本使用与问题总结

整理日期：2026-10-21  
当前学习环境：macOS、zsh，项目目录下包含带空格的 `Task 0`、`Task 1`。本文命令主要按当前环境编写。

作业依据：[Task 1 文档要求](https://github.com/5-bin/learn-AI/blob/main/tasks%282026%29/foundation/task1.md)。课程要求结合完成编程作业后的经历整理终端使用，并解释编辑器中的运行按钮。

## 一、终端、Shell、解释器各自做什么

终端提供输入命令、显示结果的界面；Shell 负责解析命令和启动程序；Python 解释器负责执行 Python 代码。

例如在项目目录输入 `python3 play.py`：Shell 先找到名为 `python3` 的可执行程序，再把 `play.py` 作为参数交给解释器。解释器读取脚本并执行。文件是否存在、当前目录是否正确、解释器是否可用，都会影响这条命令能否成功。

| 输入位置 | 可以输入什么 | 本次作业中的例子 |
| --- | --- | --- |
| Shell 命令行 | `pwd`、`cd`、`python3 ...` 等命令 | 进入项目并启动游戏 |
| Python 交互环境，通常提示 `>>>` | Python 表达式、语句 | `print("你好")` |
| 已运行程序的输入提示后 | 程序通过 `input()` 等待的数据 | 选队或技能菜单里的数字 |

如果游戏停在选择技能的提示处，它正在等待输入，此时输入 `pwd` 会被当作菜单数据。应先正常退出程序，或在终端中按 `Ctrl+C` 中断，再输入 Shell 命令。Python 交互环境可用 `exit()` 退出。

## 二、实际遇到的问题一：python 命令找不到

### 1. 现象与判断

本次在当前执行环境检查，`python3` 可用，`python` 未找到。

可在 macOS 的 Shell 中执行：

```bash
command -v python
command -v python3
python3 --version
python3 -c 'import sys; print(sys.executable)'
```

其中 `command -v python` 若没有输出且退出状态为非零，表示当前 Shell 找不到该命令。本次核对得到：

```text
python：未找到
python3：/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
版本：Python 3.13.15
```

`python` 与 `python3` 是不同的命令名。找不到前者不能直接说明系统完全没有安装 Python。Shell 按当前环境的命令搜索规则寻找程序，其中 `PATH` 指定了用于搜索可执行文件的目录。

### 2. 当前环境的处理方法

当前可以统一使用 `python3`。例如进入宝可梦项目后执行：

```bash
cd "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Pokemon_Example_副本"
python3 play.py
```

该项目需要 Python 3.10 或以上，因此既要找到命令，也要检查版本是否符合要求。如果 `python3` 也找不到，再确认是否已安装解释器、安装位置是否进入当前 Shell 的搜索路径，并在更新配置后重新打开终端。

编辑器中还要核对所选解释器。VS Code 可通过命令面板中的 `Python: Select Interpreter` 选择对应环境；随后比较编辑器运行时与终端中 `sys.executable` 的输出。如果两边不同，应先确认哪一个环境才是项目要使用的环境。环境选择见 [VS Code Python 环境说明](https://code.visualstudio.com/docs/python/environments)。

本次项目运行只使用标准库，不能把命令名找不到误判为需要安装 `pokemon`、`game` 等第三方包；这些名称对应项目内的文件。

## 三、问题二：路径问题

路径问题主要涉及三个方面：终端当前目录、传入命令的路径参数，以及程序读写文件时使用的路径基准。下面结合宝可梦、发牌和校园存档代码分别说明。

### 1. 当前目录不正确，找不到入口脚本

**现象：** 在 `5-bin` 目录直接执行 `python3 play.py`，出现文件不存在的错误，而 `play.py` 实际保存在宝可梦子目录中。

**原因：** `play.py` 是相对路径，解释器会相对于终端当前工作目录查找文件。这条命令寻找的是 `5-bin/play.py`，不会自动搜索所有子目录。

**解决方法：** 先用 `pwd` 确认当前目录，用 `ls` 查看文件，再切换到入口脚本所在目录运行：

```bash
cd "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Pokemon_Example_副本"
pwd
ls
python3 play.py
```

也可以从其他目录直接指定脚本的绝对路径：

```bash
python3 "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Pokemon_Example_副本/play.py"
```

两种方式都能确定要执行的脚本，但使用绝对脚本路径不会自动切换终端当前目录。后续读写相对路径文件时，仍需关注工作目录。

### 2. 路径包含空格，切换目录失败

**现象：** 从 `5-bin` 目录执行 `cd Task 1/Pokemon_Example_副本`，无法进入目标目录。本次此前用独立 zsh 演示时出现 `string not in pwd: Task`，具体提示可能随 Shell 而变化。

**原因：** Shell 将未加引号的空格作为参数边界，`Task 1/Pokemon_Example_副本` 因而没有作为一个完整路径传给 `cd`。

**解决方法：** 给整个路径加引号。以下命令需要从 `5-bin` 目录执行：

```bash
cd "Task 1/Pokemon_Example_副本"
pwd
ls
```

引号让 Shell 把包含空格的路径当作一个参数；路径中的中文可以保留。项目中 `Task 0`、`Task 1` 都包含空格，执行 `cd` 或传入脚本路径时应注意这一点。

### 3. 使用相对路径，输出文件位置随启动目录变化

**现象：** 程序可以启动，但从不同目录运行后，发牌文本或游戏存档出现在不同位置，查看时容易误以为文件没有生成。

**原因：** `Path("player1.txt")` 这样的相对路径默认以当前工作目录为基准，而非自动以脚本所在目录为基准。终端启动位置或编辑器运行配置改变时，工作目录也可能改变。

**解决方法：** 对于应与脚本放在一起的数据文件，明确使用脚本目录作为基准。现有发牌和校园存档模块采用的写法如下：

```python
# 项目片段：__file__ 在普通脚本或模块中指向该文件。
from pathlib import Path

output_dir = Path(__file__).parent
save_directory = Path(__file__).parent / "saves"
```

发牌文本固定写入发牌脚本所在目录，校园存档固定写入 `save_manager.py` 旁边的 `saves` 目录。需要明确的绝对基准时，可使用 `Path(__file__).resolve().parent`。

| 写法 | 表示的位置 |
| --- | --- |
| `Path.cwd()` | 当前进程的工作目录，受启动方式和运行配置影响 |
| `Path(__file__).resolve().parent` | 当前脚本或模块文件所在的目录 |
| `Path(__file__).resolve().parent / "saves"` | 当前文件旁边的 `saves` 目录 |

路径对象用 `/` 拼接目录，是 `pathlib` 的用法，见 [Python pathlib 文档](https://docs.python.org/3.13/library/pathlib.html)。交互式环境中通常没有 `__file__`，因此这段项目代码应放在脚本中使用。

## 四、三个项目的入口与运行命令

以下命令每组都从绝对目录开始，可单独执行。游戏有交互时，在对应输入提示后输入菜单编号。

娜比娅偷吃事件：

```bash
cd "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Nabia_Snack_Incident_副本"
python3 start_the_battle.py
```

校园·if 恋：

```bash
cd "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Campus_IF_Love_副本"
python3 Campus_IF_Love.py
```

宝可梦对战：

```bash
cd "/Users/bin/projects/ai-leaning_works/collection-ai-2026/foundation/5-bin/Task 1/Pokemon_Example_副本"
python3 play.py
```

拆成多个文件后，不是每个 `.py` 都是游戏入口。例如 `pokemon.py`、`skills.py` 主要定义对象，直接运行它们可能没有交互输出；这与“运行失败”不是一回事。校园入口先导入 `Game`，再创建对象并启动；宝可梦入口负责创建候选角色并启动 `Play`。

`if __name__ == "__main__":` 用来区分“作为入口直接运行”与“作为模块导入”。模块顶层代码在导入时仍会执行，因此会启动游戏的入口调用应放在合适的保护块内。

## 五、右上角的三角形运行按钮是什么意思

### 1. 可以理解为编辑器替你组织一次启动

运行至少需要确定：用哪个解释器、执行哪个脚本或模块、传入哪些参数、使用哪个工作目录，以及如何提供输入输出。三角形按钮把这些信息按编辑器当前配置交给运行系统。

对宝可梦项目，一次普通运行的核心可以理解为：

```text
工作目录：.../Task 1/Pokemon_Example_副本
解释器：项目选择的 Python 3.10+ 可执行文件
入口脚本：play.py
脚本参数：当前没有必需参数
输入输出：终端或支持交互的运行控制台
```

### 2. VS Code 的 Python 运行按钮

如果按钮提示为 `Run Python File in Terminal`，它来自 Python 扩展，会在终端中使用所选 Python 环境运行指定文件，类似执行 `python3 play.py`。运行前要确认当前文件是入口，并保存改动；工作目录以实际终端和设置为准，不能仅凭按钮图标认定。[VS Code 官方运行说明](https://code.visualstudio.com/docs/python/run)

相似的图标也可能来自其他扩展，应该通过悬停提示或下拉菜单确认命令。使用 `input()` 的作业需要能输入数据的终端或交互控制台；只显示日志的输出面板不一定提供标准输入。

### 3. PyCharm 的运行按钮

PyCharm 的运行按钮按选中的运行配置启动，配置中包括解释器、脚本或模块、工作目录、参数和环境变量等；也可以选择运行当前文件。顶部按钮可能运行已选配置，不能一概认为它总运行当前编辑的文件。[PyCharm 运行说明](https://www.jetbrains.com/help/pycharm/running-applications.html)、[Python 运行配置](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html)

### 4. 运行不等于调试，也不等于类型检查

普通运行启动程序并执行；调试运行还接入调试器，用断点、单步执行和变量查看来观察过程。调试器通常会暂停在启用的断点处，普通运行通常不会。

运行按钮执行一次程序，不代表覆盖所有输入，也不代表已经执行静态类型检查。宝可梦项目已有严格 Pyright 命令行检查记录，但这不能代替学习者实际核对 Pylance 面板中的诊断。

## 六、遇到问题时的一次检查顺序

1. 用 `command -v python3`、`python3 --version` 确认解释器命令和版本。
2. 必要时输出 `sys.executable`，核对编辑器与终端是否使用同一个解释器。
3. 用 `pwd`、`ls` 确认当前目录和入口文件；路径包含空格时，将整个路径放入引号。
4. 进入项目目录后运行入口，或明确指定入口的绝对路径；程序等待输入时，按提示输入菜单数据。
5. 输出文件位置不符合预期时，比较工作目录与脚本目录，检查读写路径的基准。
6. 若仍报错，阅读错误末行，再沿调用栈定位到自己的文件和行号，修改后保存并重新验证。

解释器、入口和路径确认无误后，再检查程序本身。例如循环导入需要梳理 `pokemon → skills → effects` 等模块依赖，项目用 `TYPE_CHECKING` 处理只供注解的导入；`ValueError` 等输入转换问题则需要检查输入校验逻辑，见 [输入校验笔记](<../文档一/Python3-输入校验与异常处理.md>)。
