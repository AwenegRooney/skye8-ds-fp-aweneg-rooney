# Stage A: MobileNetV2 TFLite Benchmark App

Flutter app for benchmarking MobileNetV2 TFLite inference on Android devices.

## Features

- Loads pre-trained MobileNetV2 model (TFLite format)
- Runs inference on sample images
- Measures inference time per image
- Displays predictions (class index, confidence score)
- Logs benchmark results to console

## Project Structure

```
stage_a_test_app/
├── pubspec.yaml              # Dependencies and assets
├── lib/
│   ├── main.dart            # App entry point
│   ├── screens/
│   │   └── benchmark_screen.dart  # Main UI with benchmark button
│   └── services/
│       └── tflite_service.dart    # TFLite model loading and inference
├── assets/
│   ├── mobilenet_v2_baseline.tflite
│   └── images/
│       ├── sample_1.jpg
│       ├── sample_2.jpg
│       └── sample_3.jpg
└── android/                 # Android-specific config
```

## Setup

### Prerequisites

- Flutter 3.0+
- Android SDK (minSdkVersion: 21)
- Android phone for testing

### Installation

```bash
cd stage_a_test_app
flutter pub get
```

### Add Model & Sample Images

1. Copy `models/mobilenet_v2_baseline.tflite` → `stage_a_test_app/assets/`
2. Add sample images → `stage_a_test_app/assets/images/`

```bash
mkdir -p stage_a_test_app/assets/images
cp models/mobilenet_v2_baseline.tflite stage_a_test_app/assets/
# Add 3-5 sample .jpg images to assets/images/
```

## Running

### Local (development)

```bash
cd stage_a_test_app
flutter run
```

### On Device via APK

GitHub Actions builds the APK automatically. Download from **Actions → Build Flutter APK → app-debug.apk**.

Then:

```bash
adb install app-debug.apk
```

## Benchmark Results

Results are logged to console and displayed in the app UI:

```
=== BENCHMARK RESULTS ===
Total inferences: 3
Avg inference time: 245.67ms
Min inference time: 210ms
Max inference time: 285ms
========================
```

These timings feed into `benchmarks/stage_a_results.json` after manual collection.

## Dependencies

- **tflite_flutter**: TFLite inference
- **image**: Image processing (resize, decode)
- **path_provider**: Access device storage