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

  // Benchmark aggregate stats
  double? _avgTime;
  int? _minTime;
  int? _maxTime;
  double? _avgRam;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      _tfliteService = TFLiteService();
      await _tfliteService.loadModel();
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Failed to load model: $e';
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _runBenchmark() async {
    setState(() {
      _isBenchmarking = true;
      _results.clear();
      _error = null;
      _avgTime = null;
      _minTime = null;
      _maxTime = null;
      _avgRam = null;
    });

    try {
      final imageAssets = [
        'assets/images/sample_1.jpeg',
        'assets/images/sample_2.jpeg',
        'assets/images/sample_3.jpeg',
      ];

      final List<int> inferenceTimings = [];
      final List<double> ramUsages = [];

      for (String asset in imageAssets) {
        // Yield to UI thread so the loading indicator keeps spinning
        await Future.microtask(() {});

        try {
          final ByteData imageData = await rootBundle.load(asset);
          final Uint8List bytes = imageData.buffer.asUint8List();

          final result = await _tfliteService.runInference(bytes);
          inferenceTimings.add(result.inferenceTimeMs);
          ramUsages.add(result.memoryMb);

          setState(() {
            _results.add(BenchmarkResult(
              imageName: asset.split('/').last,
              inferenceTimeMs: result.inferenceTimeMs,
              memoryMb: result.memoryMb,
              topClassIndex: result.topIndex,
              confidence: result.confidence,
            ));
          });
        } catch (e) {
          throw Exception('Failed on $asset. Ensure it exists in pubspec.yaml. Details: $e');
        }
      }

      // Calculate and display stats directly on the UI
      if (inferenceTimings.isNotEmpty) {
        setState(() {
          _avgTime = inferenceTimings.reduce((a, b) => a + b) / inferenceTimings.length;
          _minTime = inferenceTimings.reduce((a, b) => a < b ? a : b);
          _maxTime = inferenceTimings.reduce((a, b) => a > b ? a : b);
          _avgRam = ramUsages.reduce((a, b) => a + b) / ramUsages.length;
        });
      }

    } catch (e) {
      setState(() {
        _error = 'Benchmark failed: $e';
      });
    } finally {
      setState(() {
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
                ? Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.error, color: Colors.red, size: 48),
                        const SizedBox(height: 16),
                        Text(_error!, textAlign: TextAlign.center, style: const TextStyle(color: Colors.red)),
                        const SizedBox(height: 24),
                        ElevatedButton(
                          onPressed: _initializeModel, // Allow retry if model failed
                          child: const Text('Retry'),
                        )
                      ],
                    ),
                  )
                : SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        const Text(
                          'Stage A: TFLite Baseline Benchmark',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center,
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
                        
                        // Summary Stats Card
                        if (_avgTime != null && !_isBenchmarking)
                          Card(
                            color: Colors.blue.shade50,
                            margin: const EdgeInsets.only(bottom: 24),
                            child: Padding(
                              padding: const EdgeInsets.all(16),
                              child: Column(
                                children: [
                                  const Text('Benchmark Summary', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                                  const Divider(),
                                  Text('Avg Time: ${_avgTime!.toStringAsFixed(2)} ms', style: const TextStyle(fontWeight: FontWeight.w600)),
                                  Text('Min Time: ${_minTime} ms'),
                                  Text('Max Time: ${_maxTime} ms'),
                                  const SizedBox(height: 4),
                                  Text('Avg RAM: ${_avgRam!.toStringAsFixed(2)} MB', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue.shade800)),
                                ],
                              ),
                            ),
                          ),

                        // Individual Results List
                        if (_results.isNotEmpty)
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              const Text('Individual Results', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                              const SizedBox(height: 12),
                              ..._results.map((r) => Card(
                                    margin: const EdgeInsets.symmetric(vertical: 6),
                                    child: Padding(
                                      padding: const EdgeInsets.all(12),
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text('Image: ${r.imageName}', style: const TextStyle(fontWeight: FontWeight.bold)),
                                          const SizedBox(height: 4),
                                          Text('Inference: ${r.inferenceTimeMs}ms', style: TextStyle(color: Colors.blue.shade800)),
                                          Text('RAM Usage: ${r.memoryMb.toStringAsFixed(2)} MB'),
                                          Text('Top class index: ${r.topClassIndex}'),
                                          Text('Confidence: ${(r.confidence * 100).toStringAsFixed(2)}%'),
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
  final double memoryMb;
  final int topClassIndex;
  final double confidence;

  BenchmarkResult({
    required this.imageName,
    required this.inferenceTimeMs,
    required this.memoryMb,
    required this.topClassIndex,
    required this.confidence,
  });
}
