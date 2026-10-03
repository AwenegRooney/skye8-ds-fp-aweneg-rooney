from pathlib import Path

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ..utils.get_data import load_finetune_data, representative_dataset


def main(output_dir_loc: str = "models", epochs: int = 4):
    output_dir = Path(output_dir_loc)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights="imagenet",
        classes=1000,
    )

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.40,  # 40 % – safer than 50 %
            begin_step=0,
            end_step=200,
        )
    }

    model = tfmot.sparsity.keras.prune_low_magnitude(base, **pruning_params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    x, y = load_finetune_data(num_samples=3000)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model.fit(x, y, epochs=epochs, batch_size=8, callbacks=callbacks, verbose=1)

    # Remove pruning wrappers
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(num_calib=200)
    tflite_model = converter.convert()

    out = output_dir / "mobilenet_v2_pruned.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved Pruned → {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
