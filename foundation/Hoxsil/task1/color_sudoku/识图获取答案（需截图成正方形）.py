from PIL import Image
import subprocess
import numpy as np
import os
import cv2


FLODER_PATH = os.path.dirname(os.path.abspath(__file__))
EXE_PATH = os.path.join(FLODER_PATH, 'color_soduku.exe')
FLODER_PATH = os.path.join(FLODER_PATH, "图片丢这里")
DEBUG_IMG_PATH = os.path.join(FLODER_PATH, 'debug_image.png')


def detect_row_count(pil_img: Image.Image):
    # 获取灰度图
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 对每一行求平均亮度：白色线的行，平均值会明显更高
    row_bright = np.mean(gray, axis=1)

    # 找亮度峰值（白色横线）
    bright_thresh = 230  # 白色阈值，越白数值越接近255，可微调
    peaks = []
    for y in range(h):
        if row_bright[y] > bright_thresh:
            peaks.append(y)

    # 寻找不相邻的白色峰值个数，以统计总白线数
    row_count = 0
    gap = 5
    last_y = peaks[0]

    for y in peaks:
        if y - last_y > gap:
            row_count += 1
        last_y = y

    return row_count

def board_to_matrix(img_path, color_threshold=40):
    # 打开原图，裁剪棋盘区域
    im = Image.open(img_path).convert("RGB")

    grid_size = detect_row_count(im)

    w, h = im.size

    cell_w = int(w / grid_size)

    add_w = int(cell_w / 4)

    origin = int(w / (2 * grid_size))

    color_map = dict()   # key:基准rgb元组, value:编号
    matrix = []
    next_id = 1

    for row in range(grid_size):
        row_data = []
        for col in range(grid_size):
            # 当前格子的裁剪范围
            cx0 = origin + col * cell_w
            cy0 = origin + row * cell_w

            # 裁剪小格子，求区域RGB均值，抗噪
            cell = im.crop((cx0-add_w, cy0-add_w, cx0+add_w, cy0+add_w))
            arr = np.array(cell)
            rgb = tuple(np.mean(arr, axis=(0,1)).astype(int))

            # 匹配颜色然后编码
            match_id = None
            # 模糊匹配
            for base_rgb, cid in color_map.items():
                dr = rgb[0] - base_rgb[0]
                dg = rgb[1] - base_rgb[1]
                db = rgb[2] - base_rgb[2]
                dist = np.sqrt(dr*dr + dg*dg + db*db)
                if dist < color_threshold:
                    match_id = cid
                    break
            # 存在则标上匹配颜色对应的序号
            if match_id is not None:
                row_data.append(match_id)
            # 不存在则赋予该颜色新的序号
            else:
                color_map[rgb] = next_id
                row_data.append(next_id)
                next_id += 1
        matrix.append(row_data)

    return matrix

def matrix_to_cpp_input(matrix):
    n = len(matrix)
    # 为第一行写上矩阵的大小
    lines = [str(n)]
    for row in matrix:
        line = " ".join(str(x) for x in row)
        lines.append(line)
    return "\n".join(lines) + "\n"

def get_answer(input_data):
    p = subprocess.Popen(
        [EXE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="gbk"
    )
    stdout, stderr = p.communicate(input=input_data)
    stdout_list = stdout.split(r' \n')
    for x in stdout_list:
        print(x)


jpg_names = []
for filename in os.listdir(FLODER_PATH):
    if filename.lower().endswith(".jpg"):
        jpg_names.append(filename)

for filename in jpg_names:
    file_path = os.path.join(FLODER_PATH, filename)
    input_data = board_to_matrix(file_path, color_threshold=15)
    input_data = matrix_to_cpp_input(input_data)
    print("input_data:")
    print(input_data)
    get_answer(input_data)
    print()
