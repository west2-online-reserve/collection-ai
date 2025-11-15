from pathlib import Path
from utils import *


class Whisper:

    def __init__(self, whisper_cpp_path: Path, ggml_path: Path) -> None:
        self.whisper_cpp_path = whisper_cpp_path
        self.ggml_path = ggml_path

    def generate_subtitle_raw(self, file_path: Path, lang: str = 'zh', trans_to: str = 'None', ggml_prompt: str = '',max_char_per_sentence:str="60"):

        ggml_prompt_ = ""
        if lang == 'zh':
            ggml_prompt_ = "以下是简体中文的语音生成文字，"+ggml_prompt

        output_path = STORAGE_PATH/"subtitles_raw"/file_path.stem

        command = [
            str(self.whisper_cpp_path),
            "-m", str(self.ggml_path),
            "-f", str(file_path).strip(),
            "-osrt",
            "-l", lang,
            "--prompt", ggml_prompt_,
            "-of", str(output_path),
            "-ml", max_char_per_sentence
        ]

        command = [arg.strip() for arg in command]
        if run_command(command, __name__):
            print(f"成功导出{file_path.name}的字幕：{output_path}")

        else:
            print(f"导出{file_path.name}字幕失败")
            return None
