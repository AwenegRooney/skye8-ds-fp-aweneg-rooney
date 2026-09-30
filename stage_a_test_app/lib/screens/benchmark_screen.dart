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
    ModelConfig(name: 'Baseline (FP32)', assetPath: 'assets/models/mobilenet_v2_baseline.tflite'),
    ModelConfig(name: 'PTQ (INT8)', assetPath: 'assets/models/mobilenet_v2_ptq.tflite'),
    ModelConfig(name: 'QAT (INT8)', assetPath: 'assets/models/mobilenet_v2_qat.tflite'),
    ModelConfig(name: 'Pruned (50% Sparse)', assetPath: 'assets/models/mobilenet_v2_pruned.tflite'),
    ModelConfig(name: 'Distilled (alpha=0.35)', assetPath: 'assets/models/mobilenet_v2_distilled.tflite'),
  ];

  late ModelConfig _selectedModel;
  bool _isBenchmarking = false;
  String? _error;

  // Comparison results map: Model Name -> Aggregate Summary
  final Map<String, ModelSummary> _suiteResults = {};

  @override
  void initState() {
    super.initState();
    _selectedModel = _availableModels.first;
  }

  Future<ModelSummary?> _benchmarkSingleModel(ModelConfig model) async {
    final imageAssets = [
      'assets/images/sample_1.jpeg',
      'assets/images/sample_2.jpeg',
      'assets/images/sample_3.jpeg',
    ];

    try {
      await _tfliteService.loadModel(model.assetPath);
      final List<int> timings = [];
      final List<double> rams = [];

      for (String asset in imageAssets) {
        await Future.microtask(() {}); // Keep UI responsive
        final ByteData imageData = await rootBundle.load(asset);
        final Uint8List bytes = imageData.buffer.asUint8List();

        final result = await _tfliteService.runInference(bytes);
        timings.add(result.inferenceTimeMs);
        rams.add(result.memoryMb);
      }

      double avgTime = timings.reduce((a, b) => a + b) / timings.length;
      int minTime = timings.reduce((a, b) => a < b ? a : b);
      int maxTime = timings.reduce((a, b) => a > b ? a : b);
      double avgRam = rams.reduce((a, b) => a + b) / rams.length;

      return ModelSummary(
        modelName: model.name,
        avgTimeMs: avgTime,
        minTimeMs: minTime,
        maxTimeMs: maxTime,
        avgRamMb: avgRam,
      );
    } catch (e) {
      debugPrint('Error benchmarking ${model.name}: $e');
      return null;
    }
  }

  Future<void> _runSelectedBenchmark() async {
    setState(() {
      _isBenchmarking = true;
      _error = null;
    });

    final summary = await _benchmarkSingleModel(_selectedModel);
    setState(() {
      if (summary != null) {
        _suiteResults[_selectedModel.name] = summary;
      } else {
        _error = 'Failed to benchmark ${_selectedModel.name}. Verify asset exists in pubspec.yaml.';
      }
      _isBenchmarking = false;
    });
  }

  Future<void> _runAllBenchmarksSuite() async {
    setState(() {
      _isBenchmarking = true;
      _suiteResults.clear();
      _error = null;
    });

    for (var model in _availableModels) {
      final summary = await _benchmarkSingleModel(model);
      if (summary != null) {
        setState(() {
          _suiteResults[model.name] = summary;
        });
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
            // Model Selection Dropdown
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
                        items: _availableModels.map>((ModelConfig m) {
                          return DropdownMenuItem(
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

            // Action Buttons
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
                      backgroundColor: Colors.blue.shade(700),
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

            // Results Dashboard
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
                        'Avg Latency: ${s.avgTimeMs.toStringAsFixed(2)} ms (Min: ${s.minTimeMs}ms, Max: ${s.maxTimeMs}ms)\n'
                        'Avg Memory RSS: ${s.avgRamMb.toStringAsFixed(2)} MB',
                      ),
                      trailing: Icon(
                        s.modelName.contains('Baseline') ? Icons.data_usage : Icons.speed,
                        color: Colors.blue.shade(800),
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

  ModelSummary({
    required this.modelName,
    required this.avgTimeMs,
    required this.minTimeMs,
    required this.maxTimeMs,
    required this.avgRamMb,
  });
}