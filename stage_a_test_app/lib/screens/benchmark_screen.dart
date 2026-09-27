import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../services/tflite_service.dart';
import 'dart:typed_data';

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({Key? key}) : super(key: key);

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  late TFLiteService _tfliteService;
  bool _isLoading = true;
  bool _isBenchmarking = false;
  List<BenchmarkResult> _results = [];
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      _tfliteService = TFLiteService();
      await _tfliteService.loadModel();
      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load model: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _runBenchmark() async {
    setState(() {
      _isBenchmarking = true;
      _results = [];
    });

    try {
      // Load sample images from assets
      final imageAssets = [
        'assets/images/sample_1.jpg',
        'assets/images/sample_2.jpg',
        'assets/images/sample_3.jpg',
      ];

      final List<int> inferenceTimings = [];

      for (String asset in imageAssets) {
        try {
          final ByteData imageData = await rootBundle.load(asset);
          final Uint8List bytes = imageData.buffer.asUint8List();

          final result = await _tfliteService.runInference(bytes);
          inferenceTimings.add(result.inferenceTimeMs);

          setState(() {
            _results.add(BenchmarkResult(
              imageName: asset.split('/').last,
              inferenceTimeMs: result.inferenceTimeMs,
              topClassIndex: result.topIndex,
              confidence: result.confidence,
            ));
          });

          await Future.delayed(const Duration(milliseconds: 500));
        } catch (e) {
          print('Error processing $asset: $e');
        }
      }

      // Calculate stats
      if (inferenceTimings.isNotEmpty) {
        final avgTime = inferenceTimings.reduce((a, b) => a + b) / inferenceTimings.length;
        final minTime = inferenceTimings.reduce((a, b) => a < b ? a : b);
        final maxTime = inferenceTimings.reduce((a, b) => a > b ? a : b);

        print('=== BENCHMARK RESULTS ===');
        print('Total inferences: ${inferenceTimings.length}');
        print('Avg inference time: ${avgTime.toStringAsFixed(2)}ms');
        print('Min inference time: ${minTime}ms');
        print('Max inference time: ${maxTime}ms');
        print('========================');
      }

      setState(() {
        _isBenchmarking = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Benchmark failed: $e';
        _isBenchmarking = false;
      });
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
        title: const Text('MobileNetV2 Benchmark'),
        centerTitle: true,
      ),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator()
            : _error != null
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.error, color: Colors.red, size: 48),
                      const SizedBox(height: 16),
                      Text(_error!, textAlign: TextAlign.center),
                    ],
                  )
                : SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        const Text(
                          'Stage A: TFLite Baseline Benchmark',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 24),
                        ElevatedButton(
                          onPressed: _isBenchmarking ? null : _runBenchmark,
                          child: _isBenchmarking
                              ? const SizedBox(
                                  height: 20,
                                  width: 20,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Text('Run Benchmark'),
                        ),
                        const SizedBox(height: 24),
                        if (_results.isNotEmpty)
                          Column(
                            children: [
                              const Text(
                                'Results',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 12),
                              ..._results.map((r) => Card(
                                    margin: const EdgeInsets.symmetric(vertical: 8),
                                    child: Padding(
                                      padding: const EdgeInsets.all(12),
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            'Image: ${r.imageName}',
                                            style: const TextStyle(fontWeight: FontWeight.w600),
                                          ),
                                          const SizedBox(height: 8),
                                          Text('Inference time: ${r.inferenceTimeMs}ms'),
                                          Text('Top class: ${r.topClassIndex}'),
                                          Text(
                                            'Confidence: ${(r.confidence * 100).toStringAsFixed(2)}%',
                                          ),
                                        ],
                                      ),
                                    ),
                                  )),
                            ],
                          ),
                      ],
                    ),
                  ),
      ),
    );
  }
}

class BenchmarkResult {
  final String imageName;
  final int inferenceTimeMs;
  final int topClassIndex;
  final double confidence;

  BenchmarkResult({
    required this.imageName,
    required this.inferenceTimeMs,
    required this.topClassIndex,
    required this.confidence,
  });
}