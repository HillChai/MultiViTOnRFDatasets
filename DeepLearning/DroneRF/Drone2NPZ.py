import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def gen_2D_df_and_save(n_samples, output_dir):
    labels = ['00000', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000']
    labels_index = {label: i for i, label in enumerate(labels)}

    for drone in ['AR', 'Background', 'Bebop', 'Phantom']:
        # 创建无人机类型子目录
        drone_output_dir = Path(output_dir) / drone
        drone_output_dir.mkdir(parents=True, exist_ok=True)

        file_list = list(Path(f"/home/ccc/DroneRF/dataset/{drone}").glob("**/*.csv"))
        file_list = [file.as_posix() for file in file_list]
        sorted_file_list = sorted(file_list, key=lambda x: (
            int(re.search(r'(\d{5})', x).group(1)),
            int(re.search(r'_(\d+)\.csv', x).group(1)),
            0 if 'L' in x else 1,
        ))

        category_records = {label: [] for label in labels}

        for i in tqdm(range(0, len(sorted_file_list), 2), desc=f"Processing {drone}"):
            BUI_label = re.search(r'(\d{5})', sorted_file_list[i]).group(1)
            index = labels_index[BUI_label]

            # 加载样本数据
            samples_Lower = np.loadtxt(sorted_file_list[i], delimiter=',', dtype=np.float64)
            samples_Higher = np.loadtxt(sorted_file_list[i + 1], delimiter=',', dtype=np.float64)

            # 计算每个文件的最大有效样本数
            max_samples = min(len(samples_Lower), len(samples_Higher)) // n_samples

            for j in range(max_samples):
                sample = (np.array([samples_Lower[j * n_samples:(j + 1) * n_samples],
                                    samples_Higher[j * n_samples:(j + 1) * n_samples]])
                          .reshape(n_samples, 2, 1))
                category_records[BUI_label].append({"sample": sample, "mode": index})

        # 保存每个类别的数据到对应的无人机类型目录
        for label, records in category_records.items():
            if records:
                df = pd.DataFrame(records)
                df = df.sample(frac=1).reset_index(drop=True)
                samples = np.array([record["sample"] for record in df.to_dict("records")])
                modes = np.array([record["mode"] for record in df.to_dict("records")])
                output_path = drone_output_dir / f"{label}.npz"
                np.savez_compressed(output_path, samples=samples, modes=modes)
                print(f"Dataset for {drone} - {label} saved to {output_path}")


# 使用示例
gen_2D_df_and_save(
    n_samples=24000,
    output_dir="/home/ccc/npz/DeepLearning/DroneRF"
)

