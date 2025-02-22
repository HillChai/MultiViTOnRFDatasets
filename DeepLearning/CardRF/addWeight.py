import os
import numpy as np

# 📌 你的原始数据路径
root_dir = "/home/ccc/npz/DeepLearning/CardRF/CardRF/LOS/Train"  
# 📌 你的目标数据路径
save_dir = "/home/ccc/npz/DeepLearning/CardRF/WeightCardRF/LOS/Train"  

# 🔹 你的类别样本分布
lable_cnt = {
    20: 85400, 19: 85400, 12: 42700, 11: 42700, 10: 42700, 9: 42700, 8: 85400,
    5: 85400, 6: 42700, 7: 42700, 18: 85400, 17: 85400, 16: 85400, 14: 59780,
    15: 85400, 13: 85400, 1: 85400, 3: 85400, 0: 85400, 4: 59780, 2: 85400
}

# 📌 计算类别权重
total_samples = sum(lable_cnt.values())
class_weights = {i: max(1.0, np.sqrt(total_samples / (1.2 * count))) for i, count in lable_cnt.items()}

# 📌 确保目标文件夹结构与源文件夹一致
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 📌 处理 `.npz` 文件，增加 `sample_weight`
def add_weight_npz(npz_path, save_path):
    # 读取 npz 文件
    with np.load(npz_path) as data:
        X_data = data["X"]  # (244, 20480)
        y_data = data["y"]  # (244,)

    # 计算 `sample_weight`
    sample_weights = np.array([class_weights.get(int(label), 1.0) for label in y_data])

    # 保存到新目录
    ensure_dir_exists(os.path.dirname(save_path))
    np.savez(save_path, X=X_data, y=y_data, w=sample_weights)
    print(f"✔ 已处理: {save_path}, 添加 sample_weight")

# 📌 遍历所有 `.npz` 文件并处理
for root, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".npz"):
            npz_path = os.path.join(root, file)
            
            # 计算相对路径并在目标目录创建相同结构
            relative_path = os.path.relpath(npz_path, root_dir)
            save_path = os.path.join(save_dir, relative_path)
            
            # 处理并保存新的 `.npz`
            add_weight_npz(npz_path, save_path)

