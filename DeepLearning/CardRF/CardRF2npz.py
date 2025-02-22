import os
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def save_current_folder_to_npz(current_folder, output_dir, label, sample_size=784):
    """
    将当前文件夹中的 .mat 文件打包为 .npz 文件，并统一保存到指定文件夹。
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .mat 文件
    mat_files = [os.path.join(current_folder, file) for file in os.listdir(current_folder) if file.endswith(".mat")]
    if not mat_files:
        print(f"No .mat files found in {current_folder}.")
        return

    X_data = []
    y_data = []

    for file_path in tqdm(mat_files, desc=f"Processing files in {current_folder}"):
        try:

            # 加载信号数据
            if h5py.is_hdf5(file_path):
                with h5py.File(file_path, 'r') as mat_file:
                    signal_data = np.array(mat_file['Channel_1']['Data'][0])
            else:
                mat_data = loadmat(file_path)
                signal_data = mat_data['Channel_1']['Data'][0]

            # 切分数据，并丢弃小于 sample_size 的部分
            for start_idx in range(0, len(signal_data), sample_size):
                end_idx = start_idx + sample_size
                if end_idx > len(signal_data):
                    break  # 丢弃不足 784 的剩余数据
                X_data.append(signal_data[start_idx:end_idx])
                y_data.append(label)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # 如果数据不为空，保存为 .npz 文件
    if X_data:
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.int64)

        # 将 .npz 文件统一保存到输出目录，并以当前文件夹名命名
        output_path = os.path.join(output_dir, f"{os.path.basename(current_folder)}.npz")
        np.savez_compressed(output_path, X=X_data, y=y_data)
        print(f"Saved {output_path} with {len(X_data)} samples.")
    else:
        print(f"No valid data in folder {current_folder}.")


# 使用示例
folders = [
"BLUETOOTH/APPLE_IPAD3", "BLUETOOTH/APPLE_IPHONE6S", "BLUETOOTH/APPLE_IPHONE7", "BLUETOOTH/FITBIT_CHARGE3", "BLUETOOTH/MOTOROLA", 
"UAV/BEEBEERUN/FLYING", "UAV/DJI_INSPIRE/FLYING", "UAV/DJI_INSPIRE/VIDEOING", "UAV/DJI_M600/FLYING", 
"UAV/DJI_MAVICPRO/FLYING", "UAV/DJI_MAVICPRO/HOVERING", "UAV/DJI_PHANTOM/FLYING", "UAV/DJI_PHANTOM/HOVERING",
"UAV_Controller/3DR_IRIS", "UAV_Controller/BEEBEERUN", "UAV_Controller/DJI_INSPIRE", "UAV_Controller/DJI_M600", "UAV_Controller/DJI_MAVICPRO", "UAV_Controller/DJI_PHANTOM", 
"WIFI/CISCO_LINKSYS_E3200", "WIFI/TPLINK_TL_WR940N"
]

dic = {folder:i for i, folder in enumerate(folders)}

print(f"len(folders):{len(folders)}")

for folder in folders:
	label = dic[folder]
	print(f"{folder}:{label}")
	current_folder = "/home/ccc/CardRF/CardRF/LOS/Test/" + folder  # 当前文件夹路径
	output_folder = "/home/ccc/npz/DeepLearning/CardRF/CardRF/LOS/Test/" + folder  # 统一保存 .npz 文件的新目录
	# 保存当前文件夹的数据到指定文件夹
	save_current_folder_to_npz(current_folder, output_folder, label=label, sample_size=20480)
