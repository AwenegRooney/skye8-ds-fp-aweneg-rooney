import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

class TFLiteService {
  late Interpreter _interpreter;
  late List<int> _inputShape;
  late List<int> _outputShape;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilenet_v2_baseline.tflite');
      _inputShape = _interpreter.getInputTensor(0).shape;
      _outputShape = _interpreter.getOutputTensor(0).shape;
      print('Model loaded successfully');
      print('Input shape: $_inputShape');
      print('Output shape: $_outputShape');
    } catch (e) {
      print('Failed to load model: $e');
      rethrow;
    }
  }

  Future<InferenceResult> runInference(Uint8List imageBytes) async {
    try {
      final stopwatch = Stopwatch()..start();

      // Decode image
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('Failed to decode image');

      // Resize to model input size (224x224)
      image = img.copyResize(image, width: 224, height: 224);

      // Convert to Float32 normalized input
      var input = List<List<List<List<double>>>>.filled(
        1,
        List<List<List<double>>>.filled(
          224,
          List<List<double>>.filled(224, [0.0, 0.0, 0.0]),
        ),
      );

      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          var pixel = image.getPixelSafe(x, y);
          input[0][y][x] = [
            (pixel.r as double) / 255.0,
            (pixel.g as double) / 255.0,
            (pixel.b as double) / 255.0,
          ];
        }
      }

      // Run inference
      var output = List<List<double>>.filled(1, List<double>.filled(1001, 0.0));
      _interpreter.run(input, output);

      stopwatch.stop();
      final inferenceTimeMs = stopwatch.elapsedMilliseconds;

      // Get top prediction
      var predictions = output[0];
      int topIndex = 0;
      double topScore = predictions[0];
      for (int i = 1; i < predictions.length; i++) {
        if (predictions[i] > topScore) {
          topScore = predictions[i];
          topIndex = i;
        }
      }

      return InferenceResult(
        topIndex: topIndex,
        confidence: topScore,
        inferenceTimeMs: inferenceTimeMs,
        outputShape: _outputShape,
      );
    } catch (e) {
      print('Inference failed: $e');
      rethrow;
    }
  }

  void close() {
    _interpreter.close();
  }
}

class InferenceResult {
  final int topIndex;
  final double confidence;
  final int inferenceTimeMs;
  final List<int> outputShape;

  InferenceResult({
    required this.topIndex,
    required this.confidence,
    required this.inferenceTimeMs,
    required this.outputShape,
  });
}