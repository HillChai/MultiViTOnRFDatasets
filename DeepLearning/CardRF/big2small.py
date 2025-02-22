import os
import numpy as np

# 设定 chunk_size，即每个小 npz 里的样本数
chunk_size = 244  # 你可以根据 GPU/CPU 内存调整

# 遍历原始数据集目录
root_dir = "/home/ccc/npz/DeepLearning/CardRF/CardRF/LOS/Train"  # 修改为你的路径

def split_npz(npz_path, save_dir, chunk_size=5000):
    """拆分一个 npz 文件成多个小的 npz"""
    # 读取 npz 文件
    data = np.load(npz_path)
    keys = list(data.keys())
    
    # 获取数据数组
    all_arrays = [data[key] for key in keys]
    total_samples = len(all_arrays[0])  # 取第一个 key 的数据长度，假设所有 key 的长度相同

    # 计算需要拆分的份数
    num_splits = (total_samples + chunk_size - 1) // chunk_size  # 向上取整

    base_name = os.path.basename(npz_path).replace(".npz", "")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_samples)
        
        # 创建新的小 npz 数据
        split_data = {key: arr[start:end] for key, arr in zip(keys, all_arrays)}

        # 保存新的 npz
        save_path = os.path.join(save_dir, f"{base_name}_part{i}.npz")
        np.savez_compressed(save_path, **split_data)

        print(f"Saved {save_path} with {end-start} samples")

# 遍历所有 npz 文件并拆分
for root, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".npz"):
            npz_path = os.path.join(root, file)
            save_dir = os.path.join(root, "split")  # 在当前目录创建 split 文件夹
            split_npz(npz_path, save_dir, chunk_size)

