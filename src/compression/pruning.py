from pathlib import Path

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from ..utils.get_data import get_dataset

tf.keras.backend.clear_session()


def export_pruned():
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), weights="imagenet", classes=1000, include_preprocessing=False
    )

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.40,
            begin_step=0,
            end_step=300,
        )
    }

    def apply_pruning_to_supported_layers(layer):
        if isinstance(
            layer, tf.keras.layers.Conv2D | tf.keras.layers.DepthwiseConv2D | tf.keras.layers.Dense
        ):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    model = tf.keras.models.clone_model(base, clone_function=apply_pruning_to_supported_layers)

    for layer, base_layer in zip(model.layers, base.layers, strict=True):
        if hasattr(layer, "layer") and "PruningWrapper" in type(layer).__name__:
            layer.layer.set_weights(base_layer.get_weights())
        else:
            layer.set_weights(base_layer.get_weights())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    dataset = get_dataset(num_samples=None, batch_size=32)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    model.fit(dataset, epochs=3, steps_per_epoch=100, callbacks=callbacks, verbose=1)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out = Path("models/mobilenet_v3_small_pruned.tflite")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(tflite_model)
    print(f"Saved Pruned -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    export_pruned()
