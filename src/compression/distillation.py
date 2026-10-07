from pathlib import Path

import tensorflow as tf

from ..utils.get_data import get_dataset

tf.keras.backend.clear_session()


def export_distilled():
    teacher = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), weights="imagenet", classes=1000, include_preprocessing=False
    )
    teacher.trainable = False

    student = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        alpha=0.75,
        weights="imagenet",
        classes=1000,
        include_preprocessing=False,
    )

    optimizer = tf.keras.optimizers.Adam(1e-4)
    temperature = 3.0
    alpha = 0.5

    distillation_loss_fn = tf.keras.losses.KLDivergence()
    student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    dataset = get_dataset(num_samples=None, batch_size=32)

    @tf.function
    def train_step(images, labels):
        teacher_logits = teacher(images, training=False)
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)

        with tf.GradientTape() as tape:
            student_logits = student(images, training=True)
            student_probs = tf.nn.softmax(student_logits / temperature)

            dist_loss = distillation_loss_fn(teacher_probs, student_probs)
            student_loss = student_loss_fn(labels, student_logits)
            total_loss = (alpha * student_loss) + ((1 - alpha) * (temperature**2) * dist_loss)

        grads = tape.gradient(total_loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return total_loss, student_loss, dist_loss

    for step, (images, labels) in enumerate(dataset.take(150)):
        total_loss, student_loss, dist_loss = train_step(images, labels)
        if step % 20 == 0:
            print(
                f"Step {step}: total_loss = {float(total_loss):.4f} (class_loss = {float(student_loss):.4f}, dist_loss = {float(dist_loss):.4f})"
            )

    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out = Path("models/mobilenet_v3_small_distilled.tflite")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(tflite_model)
    print(f"Saved Distilled -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    export_distilled()
