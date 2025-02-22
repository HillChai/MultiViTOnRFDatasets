import os
import numpy as np

def print_npz_shapes(base_dir):
    cnt = 0
    # 遍历目录中的所有子目录和文件
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                try:
                    # 加载 npz 文件
                    data = np.load(file_path)
                    keys = data.files
                    X, y, w = data['X'], data['y'], data['w']
                    cnt += X.shape[0]
                    # 打印文件信息
                    print(f"File: {file_path}")
                    print(f"  Keys: {keys}")
                    print(f"  X shape: {X.shape}")
                    print(f"  y shape: {y.shape}")
                    print(f"  w shape: {w.shape}")
                    print(f"  Unique labels in y: {np.unique(y)}")
                    print(f"  Unique weights in w: {np.unique(w)}")
                    print(f" current summary: {cnt}\n")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

# 使用示例
print_npz_shapes("/home/ccc/npz/DeepLearning/CardRF/WeightCardRF/LOS/Train")

