import random
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime
import os

from MyModel import VisionTransformer
from MyLogger import logger
from MyPlots import save_report_and_confusion_matrix
from MyUtils import CheckpointManager, SaveCheckpointCallback

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_npz_data(npz_files, sampling_rate=0.2):
    records = []
    num_classes = 10  # 固定类别数为 10

    for file in npz_files:
        data = np.load(file)
        samples, labels = data['samples'], data['modes']

        # 生成独热编码
        one_hot_labels = np.eye(num_classes)[labels]  # 直接使用 labels 作为索引

        # 应用采样率
        total_samples = len(samples)
        num_selected = int(total_samples * sampling_rate)  # 计算需要保留的数据量

        if num_selected < total_samples:  # 只有在采样率 < 1.0 时才进行采样
            selected_indices = np.random.choice(total_samples, num_selected, replace=False)
            samples = samples[selected_indices]
            one_hot_labels = one_hot_labels[selected_indices]

        records.extend([{"sample": s, "mode": ohl} for s, ohl in zip(samples, one_hot_labels)])
        # Test Case        
        # break
    return pd.DataFrame(records)

def prepare_datasets(test_ratio=0.2):
    npz_files = list(Path("/DroneRF/dataset").rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files.")  # 添加调试信息
    if not npz_files:
        raise FileNotFoundError("No .npz files found in the dataset directory.")

    df = load_npz_data(npz_files)
    print("Loaded dataset shape, df.shape:", df.shape)
    print("df.iloc[0]['sample'].shape: ", df.iloc[0]['sample'].shape)
    print("df.iloc[0]['mode'].shape: ", df.iloc[0]['mode'].shape)
    print("df.iloc[0]['mode']: ", df.iloc[0]['mode'])
    if df.empty:
        raise ValueError("Loaded dataset is empty. Check the dataset files.")

    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=test_ratio / (1 - test_ratio), random_state=42)
    return train_df, val_df, test_df

def create_tf_dataset(df, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((np.stack(df["sample"]), np.stack(df["mode"])))
    ds = ds.shuffle(len(df)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train(is_old_training, checkpoint_dir, sample_size=24000, epochs=10, batch_size=64):
    train_df, val_df, test_df = prepare_datasets()
    train_ds, val_ds, test_ds = map(lambda df: create_tf_dataset(df, batch_size), [train_df, val_df, test_df])
    
    vit = VisionTransformer(
        patch_size=[(512, 2), (1024, 2), (2048, 2)],
        hidden_size=768, depth=12, num_heads=6, mlp_dim=256,
        num_classes=len(train_df["mode"].values[0]),
        sd_survival_probability=0.9,
    )
    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(from_logits=True, name="roc_auc"), tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=vit)
    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    if checkpoint_manager.manager.latest_checkpoint and is_old_training:
        logger.info(f"加载最新模型权重：{checkpoint_manager.manager.latest_checkpoint}")
        checkpoint.restore(checkpoint_manager.manager.latest_checkpoint)
    
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_accuracy_{accuracy:.4f}"),
            monitor="roc_auc", save_best_only=True, save_weights_only=True, verbose=1, mode="max"),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        SaveCheckpointCallback(checkpoint_manager),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-10, verbose=1)
    ]
    
    vit.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    test_loss, test_roc_auc, test_accuracy = vit.evaluate(test_ds)
    logger.info(f"Test Loss: {test_loss}, Test ROC AUC: {test_roc_auc}, Test Accuracy: {test_accuracy}")
    
    y_true, y_pred = [], []
    for x, y in tqdm(test_ds):
        preds = tf.argmax(vit.predict(x, verbose=0), axis=1).numpy()
        y_pred.extend(preds)
        y_true.extend(tf.argmax(y, axis=1).numpy())
    
    labels = [str(i) for i in range(10)]
    save_report_and_confusion_matrix(y_true, y_pred, labels)

if __name__ == "__main__":
    train(is_old_training=False, checkpoint_dir="./checkpoints", epochs=10, batch_size=64)

