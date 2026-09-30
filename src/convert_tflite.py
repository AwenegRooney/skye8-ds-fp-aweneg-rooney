import os

import tensorflow as tf

from src.utils.model_loader import load_compatible_keras_model

print("Loading baseline model...")
model = load_compatible_keras_model("models/mobilenet_v2_baseline.h5")

print("Converting baseline model to tensorflow lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.optimizations = []

tflite_model = converter.convert()

# Save the TFLite model
output_path = "models/mobilenet_v2_baseline.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

# Compare sizes
original_size = os.path.getsize("models/mobilenet_v2_baseline.h5") / (1024 * 1024)
tflite_size = os.path.getsize(output_path) / (1024 * 1024)
reduction = ((original_size - tflite_size) / original_size) * 100

print(f"✅ TFLite model saved: {output_path}")
print(f"📊 Original model: {original_size:.2f} MB")
print(f"📊 TFLite model:   {tflite_size:.2f} MB (reduction: {reduction:.1f}%)")
