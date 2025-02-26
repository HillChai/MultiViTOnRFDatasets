import glob

import tensorflow as tf
import os
import gc
import math
from MyModel import VisionTransformer as teacher_VisionTransformer
from Model import VisionTransformer as student_VisionTransformer
from Model import Block

from dataset import gen_dataset
from loss import DistillationLoss
from config import TRAIN_PATH, BATCH_SIZE, TEST_PATH, OLD_CLASSES, TEACHER_CKPT, NEW_CLASSES, LEARNING_RATE, ALPHA, TEMPERATURE, LABELS, train_count, val_count, final_learning_rate
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras import mixed_precision
from MyUtils import CheckpointManager, SaveCheckpointCallback
from MyPlots import save_report_and_confusion_matrix
from MyLogger import logger
import time
from tqdm import tqdm
import numpy as np

# 环境配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.config.optimizer.set_jit(True)

gc.enable()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


@tf.function(reduce_retracing=True)
def train_step(samples, one_labels, student_vit, teacher_vit, distill_loss_fn, optimizer, train_accuracy, train_auc):
    with tf.GradientTape() as tape:
        student_logits = student_vit(samples, training=True)

        # 只获取属于旧类别的数据
        old_class_mask = tf.reduce_any(one_labels[:, :OLD_CLASSES] > 0, axis=-1)
        filtered_samples = tf.boolean_mask(samples, old_class_mask)

        # 只在旧类别数据上运行教师模型
        teacher_logits = teacher_vit(filtered_samples, training=False)

        # 计算损失
        loss = distill_loss_fn(y_true=(one_labels, teacher_logits), y_pred=student_logits)

    grads = tape.gradient(loss, student_vit.trainable_variables)
    optimizer.apply_gradients(zip(grads, student_vit.trainable_variables))

    # 更新 `metrics`
    train_accuracy.update_state(one_labels, student_logits)
    train_auc.update_state(one_labels, student_logits)

    return loss


def log_to_tensorboard(writer, epoch, train_loss, train_accuracy, train_auc, val_loss, val_accuracy, val_auc):
    with writer.as_default():
        tf.summary.scalar("Loss/train", train_loss, step=epoch)
        tf.summary.scalar("Accuracy/train", train_accuracy.result(), step=epoch)
        tf.summary.scalar("AUC/train", train_auc.result(), step=epoch)

        # Log validation metrics
        tf.summary.scalar("Loss/val", val_loss, step=epoch)
        tf.summary.scalar("Accuracy/val", val_accuracy.result(), step=epoch)
        tf.summary.scalar("AUC/val", val_auc.result(), step=epoch)

        writer.flush()  # ✅ Ensure logs are written


def train(is_test_only: bool, is_old_training: bool, train_path, test_path, checkpoint_dir, log_dir, epochs):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 启用混合精度训练
    policy = mixed_precision.Policy('mixed_float16') # float32
    mixed_precision.set_global_policy(policy)

    # 加载数据
    train_ds, val_ds = gen_dataset(TRAIN_PATH, BATCH_SIZE)
    test_ds = gen_dataset(TEST_PATH, BATCH_SIZE, is_test=True)

    # 加载教师模型
    teacher_vit = teacher_VisionTransformer(
        patch_size=[(1, 512), (1, 1024), (1, 2048)],
        hidden_size=768,
        depth=12,
        num_heads=6,
        mlp_dim=256,
        num_classes=OLD_CLASSES,
        sd_survival_probability=0.9,
    )
    teacher_vit.load_weights(TEACHER_CKPT).expect_partial()
    teacher_vit.trainable = False  # 冻结教师模型

    # 创建学生模型
    student_vit = student_VisionTransformer(
        patch_size=[(1, 512), (1, 1024), (1, 2048)],
        hidden_size=512,  #512
        depth=8,   #8
        num_heads=6,
        mlp_dim=256,
        num_classes=NEW_CLASSES,
        sd_survival_probability=0.9,
        adapter_dim=128,
    )

    train_steps = math.ceil(train_count / BATCH_SIZE)
    val_steps = math.ceil(val_count / BATCH_SIZE)

    # 余弦退火学习率调度器
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=train_steps * 5,
        t_mul=2.0,
        m_mul=0.5,
        alpha=final_learning_rate
    )

    # 定义优化器和损失函数
    # 配置优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    best_models = sorted(glob.glob(checkpoint_dir+"/best_student_*.h5"))
    if is_old_training and best_models:
        latest_best_model = best_models[-1]  # 最新的 .h5 文件
        logger.info(f"📥 正在加载最佳模型: {latest_best_model}")

        # ✅ Fix: Initialize student model first
        dummy_input = tf.random.normal([1] + list(train_ds.element_spec[0].shape[1:]))  # Create a dummy input
        _ = student_vit(dummy_input, training=False)  # Run model once to create weights

        # ✅ Now load weights
        student_vit.load_weights(latest_best_model)
        logger.info("✅ 成功加载最佳模型！")

    if not is_test_only:    

        distill_loss_fn = DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)

        # 初始化 metrics
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        train_auc = tf.keras.metrics.AUC()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_auc = tf.keras.metrics.AUC()

        best_val_acc = 0.0  # 记录最佳验证集准确率

        writer = tf.summary.create_file_writer(log_dir)
        # 训练
        for epoch in range(epochs):
            epoch_loss = 0.0  # Reset loss for each epoch
            progress_bar = tqdm(range(train_steps), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch",
                                dynamic_ncols=True)  # ✅ Fix progress bar

            for step, (samples, one_labels) in zip(progress_bar, train_ds):  # ✅ Fix infinite looping
                loss = train_step(samples, one_labels, student_vit, teacher_vit, distill_loss_fn, optimizer, train_accuracy,
                                  train_auc)
                loss_value = loss.numpy()

                epoch_loss += loss_value

                # ✅ Get the current learning rate
                if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                    current_lr = optimizer.learning_rate(tf.keras.backend.get_value(optimizer.iterations)).numpy()
                else:
                    current_lr = tf.keras.backend.get_value(optimizer.learning_rate)

                # Ensure safe conversion from Tensor to NumPy or Python float
                loss_value = loss.numpy() if hasattr(loss, "numpy") else float(loss)
                accuracy_value = train_accuracy.result().numpy() if hasattr(train_accuracy.result(), "numpy") else float(
                    train_accuracy.result())
                auc_value = train_auc.result().numpy() if hasattr(train_auc.result(), "numpy") else float(
                    train_auc.result())

                # Update tqdm progress bar safely
                progress_bar.set_postfix_str(
                    f"Loss={loss_value:.2f}, Acc={accuracy_value:.4f}, AUC={auc_value:.4f}, lr={current_lr:.3e}"
                )

            # Compute average training loss
            epoch_loss /= train_steps
            logger.info(
                f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy.result().numpy():.4f}, AUC: {train_auc.result().numpy():.4f}")

            # 验证集评估
            print("🔍 Running validation...")
            val_loss = 0.0
            val_accuracy.reset_states()
            val_auc.reset_states()

            for step, (samples, labels) in tqdm(enumerate(val_ds), total=val_steps, desc="Validating", unit="batch"):
                student_logits = student_vit(samples, training=False)

                # Compute loss
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                loss = loss_fn(labels, student_logits)
                val_loss += loss.numpy()

                val_accuracy.update_state(labels, student_logits)  # ✅ Correct way
                val_auc.update_state(labels, student_logits)  # ✅ AUC expects probabilities

            # Compute average validation loss
            val_loss /= val_steps

            # Print final validation results
            logger.info(f"✅ Validation Loss: {val_loss:.4f}")
            logger.info(f"✅ Validation Accuracy: {val_accuracy.result().numpy():.4f}")
            logger.info(f"✅ Validation AUC: {val_auc.result().numpy():.4f}")

            log_to_tensorboard(writer, epoch, epoch_loss, train_accuracy, train_auc, val_loss, val_accuracy, val_auc)

            # 保存最佳模型
            if val_accuracy.result().numpy() > best_val_acc:
                best_val_acc = val_accuracy.result().numpy()
                BEST_STUDENT_CKPT = os.path.join(checkpoint_dir, f"best_student_{best_val_acc:.4f}_{int(time.time())}.h5")
                student_vit.save_weights(BEST_STUDENT_CKPT)
                logger.info(f"Best model saved with val_acc: {best_val_acc:.4f}")

            # 重置 `metrics`
            train_accuracy.reset_states()
            train_auc.reset_states()

    # 测试集评估
    logger.info("\nEvaluating on test set:")
    student_vit.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    
    test_loss, test_accuracy, test_roc_auc = student_vit.evaluate(test_ds)
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")
    logger.info(f"Test ROC AUC: {test_roc_auc}")

    # Compile the model before evaluation
    y_pred = np.argmax(student_vit.predict(test_ds, verbose=1), axis=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    save_report_and_confusion_matrix(y_true, y_pred, LABELS)    

    
    

    


if __name__ == '__main__':
    train(is_test_only=False,
          is_old_training=True,
          train_path="/CardRFDataset/CardRF/LOS/Train",
          test_path="/CardRFDataset/CardRF/LOS/Test",
          checkpoint_dir="/SaveFolders/Adapter_distill_vit/checkpoints",
          log_dir = "/SaveFolders/Adapter_distill_vit/logs/distillation",
          epochs=5)  # 15 
