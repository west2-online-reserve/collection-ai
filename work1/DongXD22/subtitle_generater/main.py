import audio_extractor
import llm_optimizer
import subtitle_generator
from utils import *
from subtitle_generator import *
from task_json_generator import *
import json


DEBUG = 0
PAUSE = 0
JUMP_TO = 4

if not DEBUG:
    JUMP_TO = -1
    PAUSE = 0

if __name__ == "__main__":

    input_path = STORAGE_PATH/"raw_videos"

    if JUMP_TO <= 0:
        generate_tasks()
        if PAUSE:
            input()

    with open(STORAGE_PATH/'tasks.json', 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    with open(HOME_PATH/'config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    api_keys = config["api_keys"]
    whisper_cpp_path = Path(config["whisper_cpp_path"])
    ggml_path = Path(config["ggml_path"])

    gemini = llm_optimizer.Gemini(api_keys)

    for f in input_path.glob("*"):
        if f.name == ".gitkeep":
            continue
        task = tasks[f.name]
        tasks_name = task["tasks_name"]

        if JUMP_TO <= 1:
            ffmpeg = audio_extractor.Ffmpeg()
            video_path = get_path_by_folder_name("raw_videos", task)
            audio_uncut_path = get_path_by_folder_name(
                "audios_uncut", task, ".wav")
            audio_path = get_path_by_folder_name("audios", task, ".wav")
            len = task["len"]

            ffmpeg.extract_audio(video_path)
            ffmpeg.cut_audio(audio_uncut_path, len)
            if PAUSE:
                input()

        if JUMP_TO <= 2:
            whisper = subtitle_generator.Whisper(whisper_cpp_path, ggml_path)
            audio_path = get_path_by_folder_name("audios", task, ".wav")
            lang = task["lang"]
            trans_to = task["trans_to"]
            ggml_prompt = task["ggml_prompt"]

            whisper.generate_subtitle_raw(
                audio_path, lang, trans_to, ggml_prompt)
            if PAUSE:
                input()

        if JUMP_TO <= 3:
            trans_to = task["trans_to"]
            llm_prompt = task["llm_prompt"]
            llm_prompt_trans = task["llm_prompt_trans"]
            sub_raw_path = get_path_by_folder_name(
                "subtitles_raw", task, ".srt")

            gemini.llm_potmizer(sub_raw_path, llm_prompt)
            if trans_to != "None":
                gemini.llm_potmizer(sub_raw_path, llm_prompt_trans, trans_to)

            if PAUSE:
                input()

    input()
    init(tasks_name)
