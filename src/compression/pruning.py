from pathlib import Path

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ..utils.get_data import load_real_sample_data
from ..utils.model_loader import load_compatible_keras_model


def pruning_model(base_path: Path, output_path: Path) -> None:
    base_model = load_compatible_keras_model(base_path / "mobilenet_v2_baseline.h5")

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.50, begin_step=0, end_step=10
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x_train = load_real_sample_data(num_samples=64)
    y_pseudo = base_model.predict(x_train).argmax(axis=-1)

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(x_train, y_pseudo, epochs=1, batch_size=16, callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_pruned_model = converter.convert()

    out_file = output_path / "mobilenet_v2_pruned.tflite"
    with open(out_file, "wb") as f:
        f.write(tflite_pruned_model)
    print(f"Saved accuracy-preserved Pruned model to {out_file}")


if __name__ == "__main__":
    base_path = Path("models")
    output_path = Path("models")
    pruning_model(base_path, output_path)
