import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:io';

class TFLiteService {
  late Interpreter _interpreter;
  late List<int> _inputShape;
  late List<int> _outputShape;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/mobilenet_v3_small_baseline.tflite');
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

      int width = _inputShape.length > 2 ? _inputShape[1] : 224;
      int height = _inputShape.length > 2 ? _inputShape[2] : 224;
      image = img.copyResize(image, width: width, height: height);

      var input = List.generate(
        1,
        (i) => List.generate(
          height,
          (j) => List.generate(
            width,
            (k) => List.filled(3, 0.0),
          ),
        ),
      );

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          var pixel = image.getPixelSafe(x, y);
          // Safely convert the integer pixel values to double for normalization
          input[0][y][x] = [
            pixel.r.toDouble() / 255.0,
            pixel.g.toDouble() / 255.0,
            pixel.b.toDouble() / 255.0,
          ];
        }
      }

      Object output;
      if (_outputShape.length == 1) {
        output = List<double>.filled(_outputShape[0], 0.0);
      } else {
        output = List.generate(
          _outputShape[0],
          (i) => List<double>.filled(_outputShape[1], 0.0),
        );
      }

      // Run inference
      _interpreter.run(input, output);

      stopwatch.stop();
      final inferenceTimeMs = stopwatch.elapsedMilliseconds;
      final memoryMb = ProcessInfo.currentRss / (1024 * 1024);

      List<double> predictions;
      if (output is List<List<double>>) {
        predictions = output[0];
      } else if (output is List<double>) {
        predictions = output;
      } else {
        throw Exception('Unexpected output tensor type');
      }

      if (predictions.isEmpty) {
        throw Exception('Predictions list returned empty from TFLite');
      }

      // Get top prediction
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
        memoryMb: memoryMb,
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
  final double memoryMb;

  InferenceResult({
    required this.topIndex,
    required this.confidence,
    required this.inferenceTimeMs,
    required this.outputShape,
    required this.memoryMb,
  });
}
