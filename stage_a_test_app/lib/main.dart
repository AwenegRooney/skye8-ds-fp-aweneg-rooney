import 'package:flutter/material.dart';
import 'screens/benchmark_screen.dart';

void main() {
  runApp(const ModelBenchmarkApp());
}

class ModelBenchmarkApp extends StatelessWidget {
  const ModelBenchmarkApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MobileNetV2 Compression Profiler',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const BenchmarkScreen(),
    );
  }
}