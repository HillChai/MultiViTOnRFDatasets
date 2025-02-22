import datetime
import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from MyLogger import logger
from MyModel import VisionTransformer
from MyPlots import save_report_and_confusion_matrix
from MyUtils import CheckpointManager, SaveCheckpointCallback, count_files_with_extension


def load_csv_samples(file_path, labels_index, sample_size, n_samples_per_csv):
    """
    Load samples from a CSV file and yield samples of shape (sample_size, 2).
    """
    label = next((key for key in labels_index if os.path.basename(file_path).startswith(key)), None)
    if label is None:
        return
    try:
        # Load the CSV data as a 2D array with two rows
        all_samples = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        if all_samples.shape[0] != 2:
            logger.error(f"File {file_path} does not contain exactly 2 rows as expected.")
            return

        for i in range(n_samples_per_csv):
            start_index = i * sample_size
            end_index = start_index + sample_size
            if end_index > all_samples.shape[1]:
                break

            # Extract a segment of size (sample_size, 2)
            sample = np.stack((all_samples[0, start_index:end_index], all_samples[1, start_index:end_index]), axis=1)
            yield sample, np.eye(len(labels_index))[labels_index[label]]
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def data_generator(all_files, labels_index, sample_size, n_samples_per_csv, is_test):
    """
    Generator function for creating samples of shape (sample_size, 2).
    """
    if is_test:
        logger.info("Generating test dataset")
        for file_path in all_files:
            yield from load_csv_samples(file_path, labels_index, sample_size, n_samples_per_csv)
    else:
        while True:
            random.shuffle(all_files)
            for file_path in all_files:
                yield from load_csv_samples(file_path, labels_index, sample_size, n_samples_per_csv)


def gen_2D_dataset(folder_path, labels_index, sample_size, n_samples_per_csv, batch_size, is_test):
    """
    Generate a 2D dataset from CSV files with samples of shape (sample_size, 4, 1).
    """
    all_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".csv")]
    print(f"共有{len(all_files)}个csv文件")

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(all_files, labels_index, sample_size, n_samples_per_csv, is_test),
        output_signature=(
            tf.TensorSpec(shape=(sample_size, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(len(labels_index),), dtype=tf.float32)
        )
    )

    dataset = dataset.map(
        lambda x, y: (
            tf.pad(
                tf.expand_dims(x, axis=-1),  # Convert to shape (sample_size, 2, 1)
                paddings=[[0, 0], [1, 1], [0, 0]]  # Add one column before and one after along axis 1
            ),
            y
        )
    )

    if is_test:
        return dataset.batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        val_count = int(len(all_files) * 0.2)
        train_dataset = dataset.skip(val_count).shuffle(buffer_size=500).repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = dataset.take(val_count).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset, val_dataset


if __name__ == '__main__':
    is_old_training = False
    checkpoint_dir = "./checkpoints"
    epochs=10
    train_path = "./new_dataset/train"
    test_path = "./new_dataset/test"
    n_samples_per_csv = 1000000 // 20480
    batch_size = 64

    labels = ['00000', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000']
    labels_index = {label: i for i, label in enumerate(labels)}
    logger.info("num_classes: %d", len(labels_index))

    train_ds, val_ds = gen_2D_dataset(
        folder_path=train_path,
        labels_index=labels_index,
        sample_size=20480,
        n_samples_per_csv=n_samples_per_csv,
        batch_size=batch_size,
        is_test=False
    )
    test_ds = gen_2D_dataset(
        folder_path=test_path,
        labels_index=labels_index,
        sample_size=20480,
        n_samples_per_csv=n_samples_per_csv,
        batch_size=batch_size,
        is_test=True
    )

    vit = VisionTransformer(
        patch_size=[(512, 2), (1024, 2), (2048, 2)],
        hidden_size=768,
        depth=12,
        num_heads=6,
        mlp_dim=256,  # 256
        num_classes=len(labels_index),
        sd_survival_probability=0.9,
    )

    optimizer = tf.keras.optimizers.Adam(0.0001)

    # 加载最新的检查点
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vit)
    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    latest_checkpoint = checkpoint_manager.manager.latest_checkpoint
    if latest_checkpoint and is_old_training:
        logger.info(f"加载最新模型权重：{latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
    else:
        logger.info("未找到已保存的模型，训练将从头开始。")

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.AUC(from_logits=True, name="roc_auc"),
        tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    ]
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 添加 TensorBoard 日志记录，使用当前时间戳
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/fit/{current_time}/"  # 动态生成日志存储路径
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    # 模型保存回调，最多保留 5 个检查点
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_accuracy_{accuracy:.4f}"),
        monitor="accuracy",
        save_best_only=True,  # 保存最佳模型
        save_weights_only=True,  # 仅保存权重
        verbose=1,
        mode="max",
    )

    # 添加回调
    save_checkpoint_callback = SaveCheckpointCallback(checkpoint_manager)
    cbs = [
        checkpoint_callback,
        tensorboard_callback,
        save_checkpoint_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # 调整学习率下降速度
            patience=1,
            min_lr=1e-10,  # 1e-9
            verbose=1
        ),
    ]

    # 设置 steps_per_epoch 和 validation_steps
    total_train_files = count_files_with_extension(train_path)
    total_test_files = count_files_with_extension(test_path)
    steps_per_epoch = (total_train_files * n_samples_per_csv) // batch_size
    validation_steps = int(total_test_files * 0.2 * n_samples_per_csv) // batch_size

    logger.info(f"total_train_files: {total_train_files} , steps_per_epoch: {steps_per_epoch}")
    logger.info(f"total_test_files: {total_test_files}, validation_steps: {validation_steps}")

    # 模型训练
    vit.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,  # 每轮训练步数
        validation_steps=validation_steps,  # 每轮验证步数
        callbacks=cbs
    )

    # 模型评估
    test_loss, test_roc_auc, test_accuracy = vit.evaluate(test_ds)
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test ROC AUC: {test_roc_auc}")
    logger.info(f"Test Accuracy: {test_accuracy}")

    # 模型评估
    logger.info("\nEvaluating on test set:")
    y_true, y_pred = [], []
    for x, y in tqdm(test_ds):
        preds = tf.argmax(vit.predict(x, verbose=0), axis=1).numpy()
        y_pred.extend(preds)
        y_true.extend(tf.argmax(y, axis=1).numpy())

    save_report_and_confusion_matrix(y_true, y_pred, labels)
