# subtitle_generater

> 一个基于 Whisper 和 Gemini 的自动字幕生成工具

## 环境要求
- 安装`ffmpeg`并添加到系统路径

## 使用方法

### 1. 安装 Google GenAI SDK

```bash
pip install -q -U google-genai
```

### 2. 配置`whisper.cpp`

仓库地址: 
- https://github.com/ggml-org/whisper.cpp
编译并下载ggml模型
### 3. 配置`config.json`

在`config.json`中输入你的`gemini_api_keys`,`whisper-cil.exe`地址与`ggml`模型地址

### 4. 配置`task_single.json`

### 5. 将视频/音频放入`storage\raw_videos`

### 6. 运行`main.py`

## 输出结果
生成的字幕保存在`storage\archives\your_task\subtitles`或`storage\archives\your_task\subtitles_translated`中