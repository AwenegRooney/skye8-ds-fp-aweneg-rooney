from pathlib import Path

import tensorflow as tf

from ..utils.get_data import load_real_sample_data
from ..utils.model_loader import load_compatible_keras_model


def knowledge_distillation(base_path: Path, output_path: Path) -> None:
    # Teacher model (full size)
    teacher_model = load_compatible_keras_model(base_path / "mobilenet_v2_baseline.h5")
    teacher_model.trainable = False

    student_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=0.35, weights=None, classes=1000
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.KLDivergence()

    x_train = load_real_sample_data(num_samples=64)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(16)

    temperature = 3.0  # Softens output probabilities to transfer dark knowledge

    @tf.function
    def train_step(images):
        teacher_logits = teacher_model(images, training=False)
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)

        with tf.GradientTape() as tape:
            student_logits = student_model(images, training=True)
            student_probs = tf.nn.softmax(student_logits / temperature)
            loss = loss_fn(teacher_probs, student_probs)

        gradients = tape.gradient(loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        return loss

    for batch in dataset:
        train_step(batch)

    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # only standard TFLite ops
    ]
    tflite_distilled_model = converter.convert()

    out_file = output_path / "mobilenet_v2_distilled.tflite"
    with open(out_file, "wb") as f:
        f.write(tflite_distilled_model)
    print(f"Saved accuracy-preserved Distilled model to {out_file}")


if __name__ == "__main__":
    base_path = Path("models")
    output_path = Path("models")
    knowledge_distillation(base_path, output_path)
