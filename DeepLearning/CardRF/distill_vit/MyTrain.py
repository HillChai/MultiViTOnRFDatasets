"""
1. train和test的repeat分离
2. train repeat时保证steps设计的合理保证一个epoch遍历所有样本
3. 流式加载数据时需要设置repeat循环，每次循环会重新shuffle一次
4. 如果直接加载所有样本并打乱，不能实现流式，对内存影响很大，折中后采取的是打乱mat文件，再按顺序加载
5. 混合精度降低了模型的显存占用
"""

import datetime
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import LearningRateScheduler
from tqdm import tqdm

from MyModel import VisionTransformer
from MyUtils import CheckpointManager, SaveCheckpointCallback
from MyPlots import save_report_and_confusion_matrix
from MyLogger import logger

import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gc
gc.enable()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"  # CPU启用XLA
tf.config.optimizer.set_jit(True)  # 启用 XLA


# 设置随机种子
random.seed(42)

# 计算数据集的类别数
#labels = [
#    0, 1,
#    5, 6, 7, 8,
#    9, 10, 11, 12,
#    13, 14, 15,
#    19
#]
labels = [
     0, 1, 2, 3, 4,
     5, 6, 7, 8,
     9, 10, 11, 12,
     13, 14, 15, 16, 17, 18,
     19, 20
    ]
labels_index = {label: i for i, label in enumerate(labels)}
lable_cnt = {20: 85400, 19: 85400, 12: 42700, 11: 42700, 10: 42700, 9: 42700, 8: 85400,
            5: 85400, 6: 42700, 7: 42700, 18: 85400, 17: 85400, 16: 85400, 14: 59780,
            15: 85400, 13: 85400, 1: 85400, 3: 85400, 0: 85400, 4: 59780, 2: 85400}
sampling_rate = 1
total_train_samples = int(sum([lable_cnt[label] for label in labels]) * sampling_rate)
val_split = 0.2
val_count = int(total_train_samples * val_split)
train_count = total_train_samples - val_count
logger.info(f"total_train_samples: {total_train_samples};val_count: {val_count}; train_count: {train_count}")


def gen_2D_dataset(folder_path, batch_size, is_test=False):
    """
    生成流式数据集，支持从 .npz 文件中按需加载数据，同时划分验证集或读取测试集。
    Args:
        folder_path (str): 数据集文件夹路径。
        batch_size (int): 每批次大小。
        is_test (bool): 是否为测试集。

    Returns:
        tf.data.Dataset 或 (tf.data.Dataset, tf.data.Dataset): 流式数据集。
    """

    all_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if
                 file.endswith(".npz")]

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

    # 获取样本的实际长度 (20480)
    sample_length = next(iter(np.load(all_files[0])["X"])).shape[0]

    # 创建完整数据集
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


def train(is_old_training: bool,
          train_path,
          test_path,
          checkpoint_dir,
          batch_size,  # 每个批次的大小
          epochs):
    """
    训练 Vision Transformer 模型，适配 .npz 数据集。
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 启用混合精度训练
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_ds, val_ds = gen_2D_dataset(train_path, batch_size, is_test=False)
    test_ds = gen_2D_dataset(test_path, batch_size, is_test=True)

    logger.info(f"labels_index: {labels_index}")

    vit = VisionTransformer(
        patch_size=[(1, 512), (1, 1024), (1, 2048)],
        hidden_size=768,
        depth=12,
        num_heads=6,
        mlp_dim=256,
        num_classes=len(labels_index),
        sd_survival_probability=0.9,
    )

    #base_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6, clipnorm=1.0)  # 1e-6
    #optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vit)
    checkpoint = tf.train.Checkpoint(model=vit)
    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

    latest_checkpoint = checkpoint_manager.manager.latest_checkpoint
    if latest_checkpoint and is_old_training:
        logger.info(f"加载最新模型权重：{latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)

        # 强制修改学习率
        #optimizer.learning_rate.assign(6e-7)  # 这里一定要修改 base_optimizer 的学习率
        #logger.info(f"重新设定学习率: {optimizer.learning_rate.numpy()}")
    else:
        logger.info("未找到已保存的模型，训练将从头开始。")

    train_steps = math.ceil(train_count / batch_size)
    total_steps = train_steps * epochs  # 总步数
    
    # 余弦退火学习率调度器
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=2e-6,  # 初始学习率
        first_decay_steps=train_steps * 5,  # 每  5个 epoch 进行一次重启
        t_mul=2.0,  # 每次重启的间隔加倍
        m_mul=0.8,  # 每次重启的最大学习率减少
        alpha=1e-10  # 最小学习率
    )
    
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)  # 2e-6
    optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(from_logits=True, name="roc_auc"),
    ]

    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{current_time}/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_accuracy_{accuracy:.4f}"),
        monitor="accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode="max",
    )

    save_checkpoint_callback = SaveCheckpointCallback(checkpoint_manager)

    # 使用 LearningRateScheduler 记录学习率
    def lr_callback(epoch, lr):
        new_lr = base_optimizer.learning_rate.numpy()
        logger.info(f"Epoch {epoch + 1}: Learning rate adjusted to {new_lr}")
        return new_lr

    cbs = [
        checkpoint_callback,
        tensorboard_callback,
        save_checkpoint_callback,
        LearningRateScheduler(lr_callback),
        #tf.keras.callbacks.ReduceLROnPlateau(
        #    monitor='val_loss',
        #    factor=0.5,   # 0.1
        #    patience=2,
        #    min_lr=1e-10,  # 1e-10
        #    verbose=1
        #),
    ]

    #train_steps = math.ceil(train_count / batch_size)
    val_steps = math.ceil(val_count / batch_size)
    logger.info(f"train_steps: {train_steps}, val_steps: {val_steps}")

    vit.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=cbs
    )
    logger.info("\nEvaluating on test set:")
    y_pred = np.argmax(vit.predict(test_ds, verbose=1), axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])


    save_report_and_confusion_matrix(y_true, y_pred, labels)

    test_loss, test_accuracy, test_roc_auc = vit.evaluate(test_ds)
    logger.info(f"Test Loss: {test_loss}")    
    logger.info(f"Test Accuracy: {test_accuracy}")
    logger.info(f"Test ROC AUC: {test_roc_auc}")


if __name__ == '__main__':
    train(is_old_training=True,
          train_path="/CardRFDataset/CardRF/LOS/Train",
          test_path="/CardRFDataset/CardRF/LOS/Test",
          checkpoint_dir="checkpoints",
          batch_size=512,  # 每个批次的大小 256
          epochs=17)  # 15 40 
