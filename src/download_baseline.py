import os

import tensorflow as tf

print("Downloading MobileNetV2 from Keras Applications...")

# Download pre-trained MobileNetV2 directly from Keras
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=True, weights="imagenet"  # Pre-trained ImageNet weights
)

# Create output directory
os.makedirs("models", exist_ok=True)

# Save the model in the Keras 3 format so it can be reloaded by the current runtime.
model_path = "models/mobilenet_v2_baseline.h5"
model.save(model_path)
print(f"✅ Model saved: {model_path}")

# Get model size
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"📊 Model size: {size_mb:.2f} MB")

# Print model info
print("📊 Model: MobileNetV2")
print("📊 Input shape: 224x224x3")
print("📊 Output classes: 1001 (ImageNet)")
