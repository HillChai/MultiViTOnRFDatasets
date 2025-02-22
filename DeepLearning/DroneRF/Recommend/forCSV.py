import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import logging

# 设置日志记录
logging.basicConfig(filename="error_log.txt", level=logging.ERROR)

# 定义数据集路径
folder_path = "../dataset/"
all_files = []

# 遍历不同类别的文件夹
for drone in ['AR', 'Background', 'Bebop', 'Phantom']:
    file_list = list(Path(folder_path + drone).glob("**/*.csv"))
    # 将路径转为字符串格式
    for i, _ in enumerate(file_list):
        file_list[i] = file_list[i].as_posix()
    # 排序文件，按照规则排序
    sorted_file_list = sorted(file_list, key=lambda x: (
        int(re.search(r'(\d{5})', x).group(1)),
        int(re.search(r'_(\d+)\.csv', x).group(1)),
        0 if 'L' in x else 1,
    ))
    # 添加到全局文件列表
    all_files.extend(sorted_file_list)

# 确保文件总数为偶数
if len(all_files) % 2 != 0:
    print("警告：文件数目为奇数，最后一个文件将被忽略。")
    all_files = all_files[:-1]  # 移除最后一个文件

# 创建输出文件夹
output_folder = "./new_dataset/"
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 遍历文件对并将数据写入新的CSV文件
for i in tqdm(range(0, len(all_files), 2), desc="处理文件对"):
    try:
        # 加载两个CSV文件
        samples_Lower = np.loadtxt(all_files[i], delimiter=',', dtype=np.float32)
        samples_Higher = np.loadtxt(all_files[i + 1], delimiter=',', dtype=np.float32)

        # 合并数据
        combined_data = np.vstack((samples_Lower, samples_Higher))

        # 获取原始文件名并去掉第 6 位字母
        original_name = Path(all_files[i]).stem
        modified_name = original_name[:5] + original_name[6:]  # 去掉第 6 位字母

        # 生成新的文件路径
        output_file = os.path.join(output_folder, f"{modified_name}.csv")

        # 写入新的CSV文件
        np.savetxt(output_file, combined_data, delimiter=',', fmt='%.6f')

    except Exception as e:
        # 记录错误日志
        logging.error(f"Error processing files {all_files[i]} and {all_files[i + 1]}: {e}")
        print(f"处理文件 {all_files[i]} 和 {all_files[i + 1]} 时出错：{e}")
        continue
