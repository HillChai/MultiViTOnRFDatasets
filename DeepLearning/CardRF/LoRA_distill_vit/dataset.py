import tensorflow as tf
import numpy as np
import os
import random
from config import labels_index, sampling_rate, val_count
from MyLogger import logger

def gen_dataset(folder_path, batch_size, is_test=False):
    all_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".npz")]

    def data_generator():
        if not is_test:
            random.shuffle(all_files)

        for file_path in all_files:
            try:
                with np.load(file_path) as data:
                    y_data = data["y"]
                    X_data = data["X"]

                file_label = int(y_data[0])
                if file_label not in labels_index:
                    continue

                mapped_label = labels_index[file_label]
    
                total_samples = len(X_data)
                num_selected = int(total_samples * sampling_rate)

                if num_selected < total_samples:
                    selected_indices = np.linspace(0, total_samples - 1, num=num_selected, dtype=int)
                    X_data = X_data[selected_indices]


                one_hot_label = np.eye(len(labels_index))[mapped_label]
                one_hot_labels = np.tile(one_hot_label, (len(X_data), 1))

                yield from zip(X_data, one_hot_labels)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
    
    sample_length = next(iter(np.load(all_files[0])["X"])).shape[0]    

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sample_length,), dtype=tf.float32),  # 确保数据形状正确
            tf.TensorSpec(shape=(len(labels_index),), dtype=tf.float32),  # 独热编码
        )
    )

    dataset = dataset.map(lambda x, y: (tf.reshape(x, [1, sample_length, 1]), y))

    if is_test:
        return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # 划分训练集和验证集
    train_dataset = dataset.skip(val_count).shuffle(buffer_size=batch_size*10).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.take(val_count).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset

