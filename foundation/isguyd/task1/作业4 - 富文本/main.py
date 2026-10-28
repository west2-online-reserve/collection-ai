from __future__ import annotations

import pathlib

import time

import json

import msvcrt

import sys

def search_label(lines: list[str], label: str, start: int) -> int:
    for i in range(start,len(lines)):
        if lines[i].strip().startswith(label):
            return i
    return -1
#寻找标签

def check_choice(lines: list[str], line: str) -> None:
    global current_line,tempA,tempB,tempEND
    if line.startswith("#A"):
        tempA = current_line
        tempB = search_label(lines,"#B",tempA)
        tempEND = search_label(lines,"#END",tempB)
        if tempB == -1 or tempEND == -1:
            raise ValueError(f"Error: 标签缺失，第{current_line + 1}行")
        while True:
            typewriter(f'选项A：{lines[tempA].strip().removeprefix("#A").strip()}\n'+
                    f'选项B：{lines[tempB].strip().removeprefix("#B").strip()}\n'+'请选择：')
            choice = input().strip().upper()
            #防小写爆炸，懒得改了就这样吧
            if choice == 'A':
                current_line = tempA
                break
            elif choice == 'B':
                current_line = tempB
                break
            else:
                typewriter("Please enter 'A' or 'B'.")
    elif line.startswith("#END") or line.startswith("#B"):
        merge = search_label(lines,"#END",current_line)
        if merge == -1:
            raise ValueError(f"Error: 找不到 #END，第{current_line + 1}行")
        current_line = merge
            #防止没读#A直接炸，而且如果条件对了那就跳转回家了
            #选项的判断

def check_route(line: str) -> str | None:
    if line.startswith("#ROUTE"):
        route = line.removeprefix("#ROUTE").strip()
        return route
    return None
#进线的判断

def save(current_line: int, route_name: str | None) -> None:
    with s.open("w", encoding = "utf-8") as f:
        json.dump({"current_line": current_line , "route": route_name}, f, ensure_ascii = False)
#json存档，其实用什么都一样了

def pause(hold: float = 1, tick: float = 0.02) -> None:
    end = time.monotonic()+hold
    while time.monotonic()<end:
        if msvcrt.kbhit():
            get_key()
            return None
        time.sleep(tick)
#显示后暂停，手敲跳过

def typewriter(text: str, delay: float = 0.1, hold: float = 0.7) -> None:
    for i, element in enumerate(text):
        if msvcrt.kbhit():
            get_key()
            print(text[i:])
            break
            #print 自带换行不需要下面的
        print(element, end = "", flush = True)
        time.sleep(delay)
    else:
        print()
    drain_keys()
    pause(hold)
#打字机

def drain_keys() -> None:
    while msvcrt.kbhit():
        get_key()
        #清缓存

def get_key() -> str:
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):     # 方向键和F键两个字节吃掉第二个
        msvcrt.getwch()
        return ""
    if ch == "\x03":               # Ctrl+C手动退
        sys.exit()
    return ch
#防缓存秒跳 替换.getwch

def loadsave(save_path: pathlib.Path) -> None:
    global route_name,current_line,p
    if save_path.exists():
        with save_path.open("r", encoding = "utf-8" ) as savefile:
            save_data = json.load(savefile)
            current_line = save_data.get("current_line", 0)
            route_name = save_data.get("route", 0)
            if route_name:
                p = Here / f"{route_name}.txt"
#读档（强行换成函数）


current_line: int = 0
tempA: int = -1
tempB: int = -1
tempEND: int = -1

Here = pathlib.Path(__file__).parent

p = Here / "scripts.txt"

s = Here / "save.json"


route_name: str | None = None

loadsave(s)
# 读档

with p.open("r",encoding="utf-8") as f:
    lines = f.readlines()

while current_line < len(lines):
    line = lines[current_line].strip()
    route = check_route(line)
    if route:
        typewriter(f"当前路线: {route}")
        p = Here / f"{route}.txt"
        route_name = route
        if not p.exists():
            raise ValueError(f"Error: 路线文件 {route}.txt 不存在。路径:{p}")
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        current_line,tempA,tempB,tempEND = 0, -1, -1, -1
        #照顾route，防止少读第0行
        continue
    check_choice(lines, line)
    #让判断结束后读档还能显示上一句
    if  not line.startswith("#") and not line == "":
        typewriter(line)
        save(current_line,route_name)
    current_line += 1
    #这里加一是为了统一choice
    #主循环结束
