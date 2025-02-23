import tensorflow as tf

class DistillationLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, temperature=2.0, old_classes=21, new_classes=24):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.old_classes = old_classes
        self.new_classes = new_classes

    @tf.function(reduce_retracing=True)
    def call(self, y_true, student_logits):
        y_true, teacher_logits = y_true

        # Compute KL divergence loss (distillation)
        soft_targets = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        soft_predictions = tf.nn.softmax(student_logits[:, :self.old_classes] / self.temperature, axis=-1)
        kl_loss = tf.keras.losses.KLDivergence()(soft_targets, soft_predictions) * (self.temperature ** 2)


        # Compute standard categorical cross-entropy loss
        ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, student_logits) 
        
        kl_loss = tf.reduce_mean(kl_loss)
        ce_loss = tf.reduce_mean(ce_loss)

        #tf.print("kl_loss:", kl_loss)
        #tf.print("ce_loss:", ce_loss)

        # Compute total loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return total_loss

