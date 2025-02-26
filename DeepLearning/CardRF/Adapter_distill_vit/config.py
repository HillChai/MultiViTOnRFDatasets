import os
import numpy as np

# 训练参数
BATCH_SIZE = 512
LEARNING_RATE = 1e-7 # 1e-5 4e-7 1e-7 2e-8
final_learning_rate = 1e-10
TEMPERATURE = 6.0  # 蒸馏温度    2
ALPHA = 0.8  # 蒸馏损失权重    0.7

# 数据路径
TRAIN_PATH = "/CardRFDataset/CardRF/LOS/Train"
TEST_PATH = "/CardRFDataset/CardRF/LOS/Test"

# 模型存储路径
CHECKPOINT_DIR = "/SaveFolders/Adapter_distill_vit/checkpoints"
TEACHER_CKPT = "/SaveFolders/Adapter_distill_vit/checkpoints/teacher/model_epoch_20_accuracy_0.8340"  # 教师模型权重
STUDENT_CKPT = "/SaveFolders/Adapter_distill_vit/checkpoints/student_vit"

# 类别数
OLD_CLASSES = 21
NEW_CLASSES = 24  # 扩展 3 个类别

# 数据集中现有的类别数，影响标签的维度
LABELS = [
     0, 1, 2, 3, 4,
     5, 6, 7, 8,
     9, 10, 11, 12,
     13, 14, 15, 16, 17, 18,
     19, 20, 21, 22, 23,
    ]
labels_index = {label: i for i, label in enumerate(LABELS)}
lable_cnt = {20: 85400, 19: 85400, 12: 42700, 11: 42700, 10: 42700, 9: 42700, 8: 85400,
            5: 85400, 6: 42700, 7: 42700, 18: 85400, 17: 85400, 16: 85400, 14: 59780,
            15: 85400, 13: 85400, 1: 85400, 3: 85400, 0: 85400, 4: 59780, 2: 85400, 
            23: 7564, 21: 8784, 22: 7076}
sampling_rate = 1
total_train_samples = int(sum([lable_cnt[label] for label in LABELS]) * sampling_rate)
val_split = 0.2
val_count = int(total_train_samples * val_split)
train_count = total_train_samples - val_count



