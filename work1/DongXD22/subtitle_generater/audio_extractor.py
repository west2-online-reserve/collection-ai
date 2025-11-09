import subprocess
from pathlib import Path
from utils import *


class Ffmpeg:

    def __init__(self) -> None:
        pass

    def extract_audio(self, file_path: Path) -> Path:
        output_path = STORAGE_PATH/'audios_uncut'/file_path.stem
        output_path = output_path.with_suffix(".wav")
        error_path = Path("")
        command = [
            "ffmpeg",
            "-y",
            "-i", str(file_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

        if run_command(command, __name__):
            print(f"转换成功！音频已保存至: {str(output_path)}")
            return output_path

        else:
            print("FFmpeg 执行失败！")
            return error_path

    def cut_audio(self, file_path: Path, len: int = -1, start_point: str = "00:00:00.000") -> None | bool:

        output_path = STORAGE_PATH/"audios"/file_path.name
        if len == -1:
            shutil.copy2(file_path, output_path)
            return

        commend = [
            "ffmpeg",
            "-y",
            "-ss", start_point,
            "-i", str(file_path),
            "-t", str(len),
            "-c", "copy",
            str(output_path)
        ]
        if run_command(commend, __name__):
            print("FFmpeg 裁切音频成功!")
        else:
            print("FFmpeg 裁切音频失败！")

    def cut_audio_fixed(self, audio_path: str, segment_minutes: float = 20.0):
        """
        按固定时长切分音频
        segment_minutes: 每段的分钟数（默认20分钟）
        """
        segment_seconds = int(segment_minutes * 60)
        audio_path = Path(audio_path)
        out_dir = STORAGE_PATH/"audio_split"

        output_pattern = out_dir / "part_%03d.wav"
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(segment_seconds),
            "-c", "copy",
            str(output_pattern)
        ]
        print(f"正在切分音频：每段 {segment_minutes} 分钟...")
        subprocess.run(cmd, check=True)
        print(f"✅ 已生成音频片段保存在：{out_dir}")
        return out_dir


if __name__ == "__main__":
    ffmpeg = Ffmpeg()
    file_path = STORAGE_PATH/'raw_videos'/"高等数学A（下）_中国大学MOOC(慕课).ts"
    aud_path: Path = ffmpeg.extract_audio(file_path)
    ffmpeg.cut_audio(aud_path, 60, "00:01:00.000")
