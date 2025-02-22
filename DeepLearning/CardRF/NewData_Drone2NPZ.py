import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OLD_CLASSES =21

def gen_2D_df_and_save(n_samples, output_dir):
    labels = ['10110'] #, '10111', '11000'
    labels_index = {label: i for i, label in enumerate(labels)}

    for drone in ['AR', 'Phantom']:
        # 创建无人机类型子目录
        drone_output_dir = Path(output_dir) / drone
        drone_output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有符合 "L" 条件的 CSV 文件
        file_list = list(Path(f"/home/ccc/DroneRF/dataset/{drone}").glob("**/*L*.csv"))
        file_list = [file.as_posix() for file in file_list]  # 转换为字符串路径

        category_records = {label: [] for label in labels}

        for file in tqdm(file_list, desc=f"Processing {drone}"):
            BUI_label_match = re.search(r'(\d{5})', file)
            if not BUI_label_match:
                continue  # 跳过文件名格式不匹配的文件
            
            BUI_label = BUI_label_match.group(1)

            # 只处理在 labels 列表中的标签
            if BUI_label not in labels_index:
                continue  # 跳过不需要的标签
            
            index = labels_index[BUI_label]+OLD_CLASSES

            # 加载样本数据
            try:
                samples_Lower = np.loadtxt(file, delimiter=',', dtype=np.float64)
                
                # 确保数据不是空的
                if samples_Lower.size == 0:
                    continue
            except Exception as e:
                print(f"⚠️ 读取文件出错: {file} -> {e}")
                continue

            # 计算每个文件的最大有效样本数
            max_samples = len(samples_Lower) // n_samples

            for j in range(max_samples):
                X = samples_Lower[j * n_samples:(j + 1) * n_samples]  # 确保 X 是 (20480,)
                category_records[BUI_label].append({"X": X, "y": index})

        # 保存每个类别的数据到对应的无人机类型目录，每个文件最多 244 个样本
        max_samples_per_file = 244
        for label, records in category_records.items():
            if not records:
                continue  # 跳过无数据的类别

            df = pd.DataFrame(records)
            df = df.sample(frac=1).reset_index(drop=True)  # 随机打乱数据
            total_samples = len(df)

            file_count = 1
            for start_idx in range(0, total_samples, max_samples_per_file):
                end_idx = min(start_idx + max_samples_per_file, total_samples)

                batch_df = df.iloc[start_idx:end_idx]
                X = np.array([record["X"] for record in batch_df.to_dict("records")])  # X 变成 (244, 20480)
                y = np.array([record["y"] for record in batch_df.to_dict("records")])

                output_path = drone_output_dir / f"{label}_{file_count}.npz"
                np.savez_compressed(output_path, X=X, y=y)
                print(f"✅ Dataset for {drone} - {label} ({file_count}) saved to {output_path} (X shape: {X.shape})")

                file_count += 1


# 使用示例
gen_2D_df_and_save(
    n_samples=20480,
    output_dir="/home/ccc/npz/Dataset/DeepLearning/CardRF/NewData"
)

