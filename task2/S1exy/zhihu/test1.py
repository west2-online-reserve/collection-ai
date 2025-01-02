import time
import sys

def print_progress_bar(iteration, total, prefix='', length=50):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    progress_bar = f'\r{prefix} |{bar}| {percent:.2f}% Complete'

    # Move cursor to the first line
    sys.stdout.write('\033[1A')  # 上移一行
    sys.stdout.write('\033[K')   # 清除行内容
    sys.stdout.write(progress_bar + '\n')  # 写入新的进度条并换行
    sys.stdout.flush()

# 初始化进度条
print_progress_bar(0, 100, prefix='进度:')

# 示例用法
total_steps = 100
for i in range(total_steps + 1):
    print_progress_bar(i, total_steps, prefix='进度:')
    # 输出日志信息
    print(f'当前步骤: {i}')
    time.sleep(0.1)

print()  # 最后换行
