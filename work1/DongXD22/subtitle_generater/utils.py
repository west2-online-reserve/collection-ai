import subprocess
import shutil
import zipfile
import os

from pathlib import Path
from datetime import datetime

HOME_PATH = Path.cwd()
STORAGE_PATH = HOME_PATH/"storage"


def run_command(command: list[str], func_name: str) -> bool:
    command_str = ' '.join(command)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore',
            shell=False
        )
        for line in process.stdout:  # type: ignore
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"成功在:\n{func_name}运行命令:\n{command_str}")
            return True
        else:
            print(
                f"\n❌ {func_name} 运行命令:\n{command_str}，\n返回码：{process.returncode}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"在:\n{func_name}运行命令失败:\n{command_str}")
        print("---错误信息---")
        print("error_code:", e.returncode)
        print(e.stderr)
        print(e.stdout)
        return False


def arch_file(arch_name: str):
    init_folders_name = [
        "audios",
        "audios_uncut",
        "raw_videos",
        "subtitles",
        "subtitles_raw",
        "subtitles_translated",
        "tasks.json",
        "task_single.json"
    ]

    now_str = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    zip_path = (STORAGE_PATH/"archives" /
                (arch_name+now_str)).with_suffix(".zip")
    new_folder_path = STORAGE_PATH/"archives"/(arch_name+now_str)

    init_folders: list[Path] = []
    for name in init_folders_name:
        init_folders.append(STORAGE_PATH/name)
    new_folder_path.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for folder_path in init_folders:

                if folder_path.is_file():
                    shutil.copy(folder_path, new_folder_path/folder_path.name)
                    arcname = folder_path.name
                    zipf.write(folder_path, arcname)
                    print(f"Adding: {folder_path} as {arcname}")
                    continue

                shutil.copytree(folder_path, new_folder_path/folder_path.name)
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(folder_path.parent)
                        zipf.write(file_path, arcname)
                        print(f"Adding: {file_path} as {arcname}")
    except Exception as e:
        os.remove(new_folder_path)
        raise e


def clear_file():
    clear_folders_name = [
        "audios",
        "audios_uncut",
        "raw_videos",
        "subtitles",
        "subtitles_raw",
        "videos_subtitled",
        "subtitles_translated",
    ]

    clear_folders: list[Path] = []
    for name in clear_folders_name:
        clear_folders.append(STORAGE_PATH/name)

    for folder_path in clear_folders:
        if folder_path.is_file():
            os.remove(folder_path)
            print("Removing:", folder_path)
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                os.remove(file_path)
                print("Removing:", file_path)


def init(arch_name: str):
    arch_file(arch_name)
    clear_file()


def get_path_by_folder_name(folder_name: str, task: dict[str:str], suffix: str = None):
    if suffix:
        return (STORAGE_PATH/folder_name/task["raw_file_name"]).with_suffix(suffix)
    return STORAGE_PATH/folder_name/task["raw_file_name"]


if __name__ == "__main__":
    init("test")
