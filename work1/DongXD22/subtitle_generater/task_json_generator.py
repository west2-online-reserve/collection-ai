import json
from utils import *
import copy


def generate_tasks():
    input_path = STORAGE_PATH/"raw_videos"
    output_path_ = STORAGE_PATH/"tasks.json"
    tasks_ = {}
    tasks_name = ""
    task_single_raw = {}

    with open(STORAGE_PATH/'task_single.json', 'r', encoding='utf-8') as f:
        task_single_raw = json.load(f)

    for f in input_path.glob("*"):

        task_single = copy.deepcopy(task_single_raw)

        llm_prompt_config = task_single["llm_prompt_config"]
        tasks_name = task_single["tasks_name"]

        llm_prompt = f"我将会发给你{llm_prompt_config["content"]}的视频的{llm_prompt_config["lang"]}字幕文件，因为自动识别所以字幕中有大量错误，请修正，删除末尾句号，句子保持时间轴不变，逐句修正，保持源语言不变，优化上下句子之间的衔接"
        llm_prompt_trans = f"我将会发给你{llm_prompt_config["content"]}的视频的{llm_prompt_config["lang"]}字幕文件，你需要将其翻译为{llm_prompt_config["trans_to"]}，因为自动识别所以字幕中有大量错误，请修正，删除末尾句号，句子保持时间轴不变，逐句修正，翻译，保证翻译后的字幕不包含原语言，优化上下句子之间的衔接"

        if "append" in llm_prompt_config:
            llm_prompt += llm_prompt_config["append"]
        if "append_trans" in llm_prompt_config:
            llm_prompt_trans += llm_prompt_config["append_trans"]

        task_single["llm_prompt"] = llm_prompt
        task_single["llm_prompt_trans"] = llm_prompt_trans
        task_single["raw_file_name"] = f.name
        tasks_[f.name] = task_single

    try:
        with open(output_path_, 'w', encoding='utf-8') as f:
            json.dump(
                tasks_,
                f,
                ensure_ascii=False,
                indent=4
            )
        print("成功生成tasks.json")
        return tasks_name
    except IOError as e:
        print(f"写入文件时发生错误: {e}")


if __name__ == "__main__":
    generate_tasks()
