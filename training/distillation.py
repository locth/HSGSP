import tensorflow as tf
from typing import List, Optional


class Distiller(tf.keras.Model):
    """
    Knowledge Distillation wrapper for Keras models.

    Trains a student to match ground-truth labels while also matching a
    softened distribution from a (frozen) teacher model.

    Total loss = alpha * student_loss(y_true, y_pred_student)
               + (1 - alpha) * (T^2) * KL(softmax_t(teacher) || softmax_t(student))

    Where softmax_t(z) applies temperature scaling.
    """

    def __init__(self,
                 student: tf.keras.Model,
                 teacher: tf.keras.Model,
                 alpha: float = 0.5,
                 temperature: float = 4.0,
                 name: Optional[str] = None):
        super().__init__(name=name or "distiller")
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.student_from_logits = False
        self.teacher_from_logits = False

    def compile(self,
                optimizer: tf.keras.optimizers.Optimizer,
                metrics: Optional[List] = None,
                student_loss_fn: Optional[tf.keras.losses.Loss] = None,
                distillation_loss_fn: Optional[tf.keras.losses.Loss] = None,
                alpha: Optional[float] = None,
                temperature: Optional[float] = None,
                student_from_logits: Optional[bool] = None,
                teacher_from_logits: Optional[bool] = None,
                **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)

        if alpha is not None:
            self.alpha = float(alpha)
        if temperature is not None:
            self.temperature = float(temperature)

        self.student_loss_fn = student_loss_fn or tf.keras.losses.CategoricalCrossentropy()
        self.distillation_loss_fn = distillation_loss_fn or tf.keras.losses.KLDivergence()

        if student_from_logits is None:
            student_from_logits = self._infer_from_logits(self.student)
        if teacher_from_logits is None:
            teacher_from_logits = self._infer_from_logits(self.teacher)

        self.student_from_logits = bool(student_from_logits)
        self.teacher_from_logits = bool(teacher_from_logits)

    @staticmethod
    def _infer_from_logits(model: tf.keras.Model) -> bool:
        if model is None or not hasattr(model, 'layers') or not model.layers:
            return False
        last = model.layers[-1]
        if isinstance(last, tf.keras.layers.Softmax):
            return False
        activation = getattr(last, 'activation', None)
        if activation is None:
            return True
        return activation != tf.keras.activations.softmax

    def _soften(self, outputs: tf.Tensor, temperature: float, from_logits: bool) -> tf.Tensor:
        """Apply temperature scaling to logits or probabilities."""
        if from_logits:
            return tf.nn.softmax(outputs / temperature, axis=-1)

        eps = tf.keras.backend.epsilon()
        probs = tf.clip_by_value(outputs, eps, 1.0)
        log_probs = tf.math.log(probs)
        return tf.nn.softmax(log_probs / temperature, axis=-1)

    def train_step(self, data):
        x, y_true = data

        # Forward pass through student and teacher
        with tf.GradientTape() as tape:
            y_student = self.student(x, training=True)
            y_teacher = self.teacher(x, training=False)

            # Student supervised loss (with labels)
            student_loss = self.student_loss_fn(y_true, y_student)

            # Distillation loss with temperature
            T = self.temperature
            p_teacher_t = self._soften(y_teacher, T, self.teacher_from_logits)
            p_student_t = self._soften(y_student, T, self.student_from_logits)
            distill_loss = self.distillation_loss_fn(p_teacher_t, p_student_t)
            distill_loss *= (T * T)  # standard KD scaling

            if self.student.losses:
                reg_loss = tf.add_n(self.student.losses)
            else:
                reg_loss = tf.constant(0.0, dtype=student_loss.dtype)

            total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss + reg_loss

        # Backprop on student only
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics configured via compile(..., metrics=[...])
        self.compiled_metrics.update_state(y_true, y_student)
        metrics = {m.name: m.result() for m in self.metrics}

        # Log individual losses for monitoring
        metrics.update({
            'loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distill_loss,
            'reg_loss': reg_loss,
        })
        return metrics

    def test_step(self, data):
        x, y_true = data
        y_student = self.student(x, training=False)
        y_teacher = self.teacher(x, training=False)

        student_loss = self.student_loss_fn(y_true, y_student)
        T = self.temperature
        p_teacher_t = self._soften(y_teacher, T, self.teacher_from_logits)
        p_student_t = self._soften(y_student, T, self.student_from_logits)
        distill_loss = self.distillation_loss_fn(p_teacher_t, p_student_t) * (T * T)

        if self.student.losses:
            reg_loss = tf.add_n(self.student.losses)
        else:
            reg_loss = tf.constant(0.0, dtype=student_loss.dtype)

        total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss + reg_loss

        self.compiled_metrics.update_state(y_true, y_student)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            'loss': total_loss,
            'student_loss': student_loss,
            'distillation_loss': distill_loss,
            'reg_loss': reg_loss,
        })
        return metrics

    # Expose student call/predict to behave like a plain model when needed
    def call(self, inputs, training=False):
        return self.student(inputs, training=training)
