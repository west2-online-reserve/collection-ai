import json
import os
autosave_path = os.path.join(os.getcwd(),'save','autosave.json')
current_data = {'character': None, 'affinity': 0, 'story_progress': '1.1', 'choice':{}}
#data = {'character': '学姐', 'affinity': 25, 'story_progress': '1.1','choice':{ '1.1':'A', '1.2':'B', '1.3':'A'},'join_club':True}

def quicksave():
    savegame(current_data)


def loadgame(filename:str=autosave_path):
        with open(filename, "r",encoding='utf-8') as f:
            data = json.loads(f.read())
            return data

def savegame(data:dict,filename:str=autosave_path):
        with open(filename, "w",encoding='utf-8') as f:
            f.write(json.dumps(data,ensure_ascii=False)) #默认会使用ascii编码



