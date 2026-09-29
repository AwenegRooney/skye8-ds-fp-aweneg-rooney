# Stage A: MobileNetV2 TFLite Baseline

## Overview
Established baseline performance for mobilenetv2 on-device inference using TFLite format with **no optimization** applied. This serves as the reference point for measuring improvements from compression techniques in Stage B.

## Model Details
- **Model**: MobileNetV2 (ImageNet pre-trained)
- **Format**: TensorFlow Lite (.tflite)
- **Size**: 13.35 MB
- **Optimization**: None (baseline)
- **Target Classes**: ImageNet 1000 classes

## Test Environment
- **Device**: Huawei nova 3
- **OS**: Android
- **Test Date**: 2026-09-27
- **Number of Inferences**: 3 (test images)

## Performance Results

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Avg Inference Time** | 428.00ms |
| **Min Inference Time** | 384 ms |
| **Max Inference Time** | 505 ms |
| **Avg Ram Usage** | 178.41 MB |
| **Min Ram Usage** | 153.32 MB |
| **Max Ram Usage** | 198.75 MB |
| **Model Size** | 13.35 MB |

- **See stage_a_results.json for individual inference results.**

---

## Bamenda User Targets & Defense

### Context
Bamenda users use budget smartphones (Tecno Pop 7, Huawei Nova 3, Itel) with limited storage and ram.

### Target Metrics

| Metric | Baseline | Target | Defense |
|--------|----------|--------|---------|
| **App Size** | 34.96 MB (arm64-v8a) | ≤50 MB | App must fit on most budget phones |
| **Inference Time** | 428 ms | ≤500 ms | <500ms feels instant and as such users get results instantly |
| **Peak RAM** | 178 MB | ≤200 MB | Most budget phones have limited RAM |
| **Model Accuracy** | 66.67% | ≥80% | If accuracy is below 80%, users may lose trust in the app |

### Current Status
✅ **App size**: 34.96 MB (arm64-v8a) — **already meets 50 MB target**  
✅ **Inference time**: 428 ms — meets 500 ms target  
✅ **Peak RAM**: 178 MB — meets 200 MB target  
⚠ **Accuracy**: 66.67% — **5 points below 80% target**