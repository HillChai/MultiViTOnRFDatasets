import tensorflow as tf
import os
import glob

from MyLogger import logger


# 优化后的检查点管理
class CheckpointManager:
    def __init__(self, checkpoint, directory, max_to_keep=5):
        self.manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep)

    def save(self):
        self.manager.save()

    def cleanup(self):
        pass  # 清理由 CheckpointManager 自动管理


class SaveCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.save()
        logger.info(f"Checkpoint saved at the end of epoch {epoch + 1}")


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/*.index"), key=os.path.getmtime, reverse=True)
    if checkpoints:
        return checkpoints[0].replace(".index", "")
    return None


def check_dataset_shape(dataset):
    # Sample batch shape: (64, 20480, 4, 1)
    # Label batch shape: (64, 10)

    for sample, label in dataset.take(1):  # Iterates over one batch from the dataset
        print(f"Sample batch shape: {sample.shape}")
        print(f"Label batch shape: {label.shape}")


def count_files_with_extension(directory, extension=".csv"):
    """
    统计指定目录及子目录中指定扩展名的文件数量。
    Args:
        directory (str): 文件夹路径。
        extension (str): 文件扩展名。
    Returns:
        int: 文件数量。
    """
    return sum(1 for root, _, files in os.walk(directory) for file in files if file.endswith(extension))
