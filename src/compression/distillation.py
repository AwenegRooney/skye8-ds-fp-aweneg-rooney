from pathlib import Path

import tensorflow as tf

from ..utils.get_data import load_finetune_data


def main(output_dir_loc: str = "models", steps: int = 150):
    output_dir = Path(output_dir_loc)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), weights="imagenet", classes=1000
    )
    teacher.trainable = False

    student = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=0.35,
        weights="imagenet",  # critical – do NOT use weights=None
        classes=1000,
    )

    optimizer = tf.keras.optimizers.Adam(1e-4)
    temperature = 3.0
    loss_fn = tf.keras.losses.KLDivergence()

    x, _ = load_finetune_data(num_samples=64)
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(8).repeat()

    @tf.function
    def train_step(images):
        teacher_logits = teacher(images, training=False)
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)

        with tf.GradientTape() as tape:
            student_logits = student(images, training=True)
            student_probs = tf.nn.softmax(student_logits / temperature)
            loss = loss_fn(teacher_probs, student_probs)

        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    for step, batch in enumerate(dataset.take(steps)):
        loss = train_step(batch)
        if step % 20 == 0:
            print(f"Step {step}: loss = {float(loss):.4f}")

    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out = output_dir / "mobilenet_v2_distilled.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved Distilled → {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
