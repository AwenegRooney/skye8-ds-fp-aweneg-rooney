import os

import tensorflow as tf

print("Downloading MobileNetV2 from Keras Applications...")

# Download pre-trained MobileNetV2 directly from Keras
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=True, weights="imagenet"  # Pre-trained ImageNet weights
)

# Create output directory
os.makedirs("models", exist_ok=True)

# Save the model
model.save("models/mobilenet_v2_baseline.h5")
print("✅ Model saved: models/mobilenet_v2_baseline.h5")

# Get model size
size_mb = os.path.getsize("models/mobilenet_v2_baseline.h5") / (1024 * 1024)
print(f"📊 Model size: {size_mb:.2f} MB")

# Print model info
print("📊 Model: MobileNetV2")
print("📊 Input shape: 224x224x3")
print("📊 Output classes: 1001 (ImageNet)")
