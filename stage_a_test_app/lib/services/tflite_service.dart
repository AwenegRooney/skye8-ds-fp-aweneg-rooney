import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:io';

class TFLiteService {
  Interpreter? _interpreter;
  List<int> _inputShape = [];
  List<int> _outputShape = [];
  TensorType _inputType = TensorType.float32;
  TensorType _outputType = TensorType.float32;

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel(String modelPath) async {
    close();
    try {
      final interpreter = await Interpreter.fromAsset(modelPath);
      _interpreter = interpreter;

      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);

      _inputShape = inputTensor.shape;
      _outputShape = outputTensor.shape;
      _inputType = inputTensor.type;
      _outputType = outputTensor.type;
    } catch (e) {
      print('Failed to load model at $modelPath: $e');
      rethrow;
    }
  }

  Future<InferenceResult> runInference(Uint8List imageBytes) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not loaded. Call loadModel() first.');
    }

    try {
      final stopwatch = Stopwatch()..start();

      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('Failed to decode image');

      int width = _inputShape.length > 2 ? _inputShape[2] : 224;
      int height = _inputShape.length > 2 ? _inputShape[1] : 224;
      image = img.copyResize(image, width: width, height: height);

      final inputTensor = _interpreter!.getInputTensor(0);
      final double inputScale = inputTensor.params.scale;
      final int inputZeroPoint = inputTensor.params.zeroPoint;

      Object input;

      if (_inputType == TensorType.uint8) {
        input = List.generate(
          1,
          (i) => List.generate(
            height,
            (j) => List.generate(
              width,
              (k) {
                var pixel = image!.getPixelSafe(k, j); // Fixed (x, y) order
                return [pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt()];
              },
            ),
          ),
        );
      } else if (_inputType == TensorType.int8) {
        input = List.generate(
          1,
          (i) => List.generate(
            height,
            (j) => List.generate(
              width,
              (k) {
                var pixel = image!.getPixelSafe(k, j); // Fixed (x, y) order

                // 1. Normalize RGB [0, 255] -> [-1.0, 1.0]
                double normR = (pixel.r.toDouble() / 127.5) - 1.0;
                double normG = (pixel.g.toDouble() / 127.5) - 1.0;
                double normB = (pixel.b.toDouble() / 127.5) - 1.0;

                // 2. Quantize to INT8
                int quantR = (normR / inputScale + inputZeroPoint).round().clamp(-128, 127);
                int quantG = (normG / inputScale + inputZeroPoint).round().clamp(-128, 127);
                int quantB = (normB / inputScale + inputZeroPoint).round().clamp(-128, 127);

                return [quantR, quantG, quantB];
              },
            ),
          ),
        );
      } else {
        // FLOAT32 -> [-1.0, 1.0]
        input = List.generate(
          1,
          (i) => List.generate(
            height,
            (j) => List.generate(
              width,
              (k) {
                var pixel = image!.getPixelSafe(k, j); // Fixed (x, y) order
                return [
                  (pixel.r.toDouble() / 127.5) - 1.0,
                  (pixel.g.toDouble() / 127.5) - 1.0,
                  (pixel.b.toDouble() / 127.5) - 1.0,
                ];
              },
            ),
          ),
        );
      }

      Object output;
      int numClasses = _outputShape.last;
      if (_outputType == TensorType.uint8 || _outputType == TensorType.int8) {
        output = List.generate(1, (_) => List.filled(numClasses, 0));
      } else {
        output = List.generate(1, (_) => List.filled(numClasses, 0.0));
      }

      _interpreter!.run(input, output);
      stopwatch.stop();

      final outputTensor = _interpreter!.getOutputTensor(0);
      final double outputScale = outputTensor.params.scale;
      final int outputZeroPoint = outputTensor.params.zeroPoint;

      List<double> predictions = [];
      if (_outputType == TensorType.uint8 || _outputType == TensorType.int8) {
        List<dynamic> rawOutput = (output as List).first;
        predictions = rawOutput
            .map((val) => ((val as num).toDouble() - outputZeroPoint) * outputScale)
            .toList();
      } else {
        predictions = (output as List).first.cast<double>();
      }

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
        inferenceTimeMs: stopwatch.elapsedMilliseconds,
        outputShape: _outputShape,
        memoryMb: ProcessInfo.currentRss / (1024 * 1024),
      );
    } catch (e) {
      print('Inference failed: $e');
      rethrow;
    }
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
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