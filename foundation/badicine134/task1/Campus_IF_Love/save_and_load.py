import json
import os

def save(current_target,affinity,talk_turn,give_gift_turn,filename="save.json"):
    data = {
        'current_target': current_target,
        'affinity': affinity,
        'talk_turn': talk_turn,
        'give_gift_turn': give_gift_turn
    }
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
    except Exception as e:
        print(f'\n【系统提示】存档失败：{e}') # 接收异常部分由ai教学


def load(filename="save.json"):
    if not os.path.exists(filename): # 检查文件是否存在同样由ai教学
        print('\n【系统提示】未找到存档文件！')
        return None

    try:
        with open(filename,'r',encoding='utf-8') as f:
            data = json.load(f)
        print('\n【系统提示】读取存档成功')
        return data
    except Exception as e:
        print(f'【系统提示】读档失败，文件可能已损坏：{e}')
        return None