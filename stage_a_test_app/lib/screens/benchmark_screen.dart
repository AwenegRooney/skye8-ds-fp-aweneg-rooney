import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:typed_data';
import '../services/tflite_service.dart';

class ModelConfig {
  final String name;
  final String assetPath;
  const ModelConfig({required this.name, required this.assetPath});
}

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({Key? key}) : super(key: key);

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  final TFLiteService _tfliteService = TFLiteService();

  final List<ModelConfig> _availableModels = const [
    ModelConfig(name: 'Baseline (FP32)', assetPath: 'assets/models/mobilenet_v3_small_baseline.tflite'),
    ModelConfig(name: 'PTQ (INT8)', assetPath: 'assets/models/mobilenet_v3_small_ptq.tflite'),
    ModelConfig(name: 'QAT (INT8)', assetPath: 'assets/models/mobilenet_v3_small_qat.tflite'),
    ModelConfig(name: 'Pruned (50% Sparse)', assetPath: 'assets/models/mobilenet_v3_small_pruned.tflite'),
    ModelConfig(name: 'Distilled (alpha=0.35)', assetPath: 'assets/models/mobilenet_v3_small_distilled.tflite'),
  ];

  // Ground-truth ImageNet class indices for the 3 test images
  // sample_1 → Egyptian_cat (closest to Pallas's cat)
  // sample_2 → tiger
  // sample_3 → timber_wolf
  static const List<int> _groundTruth = [
    292, // tiger
    269, // timber_wolf
    281, // cat
    207, // golden_retriever
    817, // sports_car
    404, // airliner
    954, // banana
    963, // pizza
    779, // school_bus
    437, // lighthouse
  ];

  static const List<String> _imageAssets = [
    'assets/images/sample_01_tiger.jpeg',
    'assets/images/sample_02_wolf.jpeg',
    'assets/images/sample_03_cat.jpeg',
    'assets/images/sample_04_retriever.jpeg',
    'assets/images/sample_05_car.jpeg',
    'assets/images/sample_06_airliner.jpeg',
    'assets/images/sample_07_banana.jpeg',
    'assets/images/sample_08_pizza.jpeg',
    'assets/images/sample_09_bus.jpeg',
    'assets/images/sample_10_lighthouse.jpeg',
  ];

  late ModelConfig _selectedModel;
  bool _isBenchmarking = false;
  String? _error;
  final Map<String, ModelSummary> _suiteResults = {};

  @override
  void initState() {
    super.initState();
    _selectedModel = _availableModels.first;
  }

  Future<ModelSummary?> _benchmarkSingleModel(ModelConfig model) async {
    try {
      await _tfliteService.loadModel(model.assetPath);

      final List<int> timings = [];
      final List<double> rams = [];
      final List<int> predictedIndices = [];
      final List<double> confidences = [];
      int correct = 0;

      for (int i = 0; i < _imageAssets.length; i++) {
        await Future.microtask(() {}); // keep UI responsive
        final ByteData imageData = await rootBundle.load(_imageAssets[i]);
        final Uint8List bytes = imageData.buffer.asUint8List();

        final result = await _tfliteService.runInference(bytes);

        timings.add(result.inferenceTimeMs);
        rams.add(result.memoryMb);
        predictedIndices.add(result.topIndex);
        confidences.add(result.confidence);

        if (result.topIndex == _groundTruth[i]) {
          correct++;
        }
      }

      final double avgTime = timings.reduce((a, b) => a + b) / timings.length;
      final int minTime = timings.reduce((a, b) => a < b ? a : b);
      final int maxTime = timings.reduce((a, b) => a > b ? a : b);
      final double avgRam = rams.reduce((a, b) => a + b) / rams.length;
      final double accuracy = correct / _imageAssets.length; // 0.0 – 1.0

      return ModelSummary(
        modelName: model.name,
        avgTimeMs: avgTime,
        minTimeMs: minTime,
        maxTimeMs: maxTime,
        avgRamMb: avgRam,
        accuracy: accuracy,
        correctCount: correct,
        totalImages: _imageAssets.length,
        predictedIndices: predictedIndices,
        confidences: confidences,
      );
    } catch (e, stack) {
      debugPrint('Error benchmarking ${model.name}: $e');
      debugPrint(stack.toString());
      throw Exception('${model.name}: $e');
    }
  }

  Future<void> _runSelectedBenchmark() async {
    setState(() {
      _isBenchmarking = true;
      _error = null;
    });

    try {
      final summary = await _benchmarkSingleModel(_selectedModel);
      setState(() {
        if (summary != null) {
          _suiteResults[_selectedModel.name] = summary;
        }
        _isBenchmarking = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isBenchmarking = false;
      });
    }
  }

  Future<void> _runAllBenchmarksSuite() async {
    setState(() {
      _isBenchmarking = true;
      _suiteResults.clear();
      _error = null;
    });

    for (var model in _availableModels) {
      try {
        final summary = await _benchmarkSingleModel(model);
        if (summary != null) {
          setState(() {
            _suiteResults[model.name] = summary;
          });
        }
      } catch (e) {
        setState(() {
          _error = e.toString();
        });
        // continue with next model
      }
    }

    setState(() {
      _isBenchmarking = false;
    });
  }

  void _handleModelChanged(ModelConfig? val) {
    if (val != null) {
      setState(() => _selectedModel = val);
    }
  }

  @override
  void dispose() {
    _tfliteService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MobileNetV2 Metric Profiler'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: Row(
                  children: [
                    const Text('Model: ', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(width: 12),
                    Expanded(
                      child: DropdownButton<ModelConfig>(
                        value: _selectedModel,
                        isExpanded: true,
                        underline: const SizedBox(),
                        items: _availableModels.map((ModelConfig m) {
                          return DropdownMenuItem<ModelConfig>(
                            value: m,
                            child: Text(m.name),
                          );
                        }).toList(),
                        onChanged: _isBenchmarking
                            ? null
                            : (ModelConfig? val) {
                                if (val != null) {
                                  setState(() => _selectedModel = val);
                                }
                              },
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isBenchmarking ? null : _runSelectedBenchmark,
                    child: const Text('Run Selected'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue.shade700,
                      foregroundColor: Colors.white,
                    ),
                    onPressed: _isBenchmarking ? null : _runAllBenchmarksSuite,
                    child: const Text('Benchmark All'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            if (_isBenchmarking)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24.0),
                  child: CircularProgressIndicator(),
                ),
              ),

            if (_error != null)
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),

            if (_suiteResults.isNotEmpty && !_isBenchmarking) ...[
              const Text(
                'Performance Comparison Dashboard',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              ..._suiteResults.values.map((s) => Card(
                    margin: const EdgeInsets.symmetric(vertical: 6),
                    child: ListTile(
                      title: Text(s.modelName, style: const TextStyle(fontWeight: FontWeight.bold)),
                      subtitle: Text(
                        'Avg Latency: ${s.avgTimeMs.toStringAsFixed(2)} ms '
                        '(Min: ${s.minTimeMs}ms, Max: ${s.maxTimeMs}ms)\n'
                        'Avg Memory RSS: ${s.avgRamMb.toStringAsFixed(2)} MB\n'
                        'Top-1 Accuracy: ${(s.accuracy * 100).toStringAsFixed(1)}% '
                        '(${s.correctCount}/${s.totalImages})\n'
                        'Predictions: ${s.predictedIndices.join(", ")}',
                      ),
                      isThreeLine: true,
                      trailing: Icon(
                        s.modelName.contains('Baseline') ? Icons.data_usage : Icons.speed,
                        color: Colors.blue.shade800,
                      ),
                    ),
                  )),
            ],
          ],
        ),
      ),
    );
  }
}

class ModelSummary {
  final String modelName;
  final double avgTimeMs;
  final int minTimeMs;
  final int maxTimeMs;
  final double avgRamMb;
  final double accuracy;          // 0.0 – 1.0
  final int correctCount;
  final int totalImages;
  final List<int> predictedIndices;
  final List<double> confidences;

  ModelSummary({
    required this.modelName,
    required this.avgTimeMs,
    required this.minTimeMs,
    required this.maxTimeMs,
    required this.avgRamMb,
    required this.accuracy,
    required this.correctCount,
    required this.totalImages,
    required this.predictedIndices,
    required this.confidences,
  });
}