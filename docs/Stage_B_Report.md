# Stage B – Model Compression Report

**Project:** On-Device Model Serving Layer  
**Model:** MobileNetV3-Small  
**Device:** Huawei nova 3 
**Test set:** 10 labelled ImageNet-style images  

---

## 1. Techniques compared

| # | Technique | Description |
|---|-----------|-------------|
| 1 | **Baseline (FP32)** | Official ImageNet weights, no compression |
| 2 | **Post-Training Quantization (PTQ)** | Full integer INT8 with representative calibration set |
| 3 | **Magnitude Pruning** | ~40% unstructured sparsity + short fine-tuning, then TFLite export |
| 4 | **Knowledge Distillation** | Teacher = full MobileNetV3-Small; student = α=0.75; KL + classification loss |

---

## 2. Model files

Save under:

```text
models/
├── mobilenet_v3_small_baseline.tflite   (10.2 MB)
├── mobilenet_v3_small_ptq.tflite        ( 2.9 MB)
├── mobilenet_v3_small_pruned.tflite     ( 2.8 MB)
└── mobilenet_v3_small_distilled.tflite  ( 2.2 MB)
```

---

## 3. Results table (four numbers per technique)

| Technique | Size on disk | Avg memory (RSS) | Avg latency (phone) | Top-1 accuracy |
|-----------|--------------|------------------|---------------------|----------------|
| **Baseline (FP32)** | 10.2 MB | 181.26 MB | 382 ms | **80% (8/10)** |
| **PTQ (INT8)** | 2.9 MB | **141.37 MB** | **207 ms** | 40% (4/10) |
| **Pruned (~40% sparse)** | 2.8 MB | 187.17 MB | 345 ms | 50% (5/10) |
| **Distilled (α=0.75)** | **2.2 MB** | 186.45 MB | 343 ms | **80% (8/10)** |

Measurements were taken on a real phone using the Flutter + `tflite_flutter` benchmark app (average over the 10 test images).

---

## 4. Where accuracy is lost

| Technique | Accuracy change | Observation |
|-----------|-----------------|-------------|
| **PTQ** | 80% → 40% | Largest drop. Full integer quantisation reduces activation precision. |
| **Pruning** | 80% → 50% | Moderate drop. Removing ~40% of weights reduces capacity; short fine-tuning recovers part of the accuracy. Failures are spread across the set rather than only “rare” classes. |
| **Distillation** | 80% → 80% | No drop on this 10-image set. The narrower student (α=0.75) trained with temperature-scaled KL divergence plus classification loss matched baseline top-1 while cutting size to 2.2 MB. |

---

## 5. Trade-off summary

- **Smallest model:** Distilled — 2.2 MB, same accuracy as baseline on this set.  
- **Fastest / lowest RAM:** PTQ — 275 ms, 141 MB (largest accuracy cost).  
- **Balanced compression:** Pruned — 2.8 MB at 50% accuracy.  
- **Reference:** Baseline — 10.2 MB, 80% accuracy, highest latency.

---

## 6. Stage B conclusion

We compressed MobileNetV3-Small with post-training quantisation (PTQ), magnitude pruning (~40% sparsity), and knowledge distillation (student width α=0.75). On-device measurements on 10 labelled images show:

- Baseline accuracy **80%** at 10.2 MB / 497 ms  
- PTQ reduces size to **2.9 MB** and latency to **207 ms**, but accuracy falls to **40%**  
- Pruning reaches **2.8 MB** at **50%** accuracy  
- Distillation reaches **2.2 MB** while matching baseline accuracy (**80%**) on this set  
