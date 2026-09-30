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

  /// Load any TFLite model dynamically by asset path
  Future<void> loadModel(String modelPath) async {
    close(); // Close existing interpreter if open
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      _inputShape = inputTensor.shape;
      _outputShape = outputTensor.shape;
      _inputType = inputTensor.type;
      _outputType = outputTensor.type;

      print('Model loaded: $modelPath');
      print('Input shape: $_inputShape, type: $_inputType');
      print('Output shape: $_outputShape, type: $_outputType');
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

      // Decode image
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('Failed to decode image');

      int width = _inputShape.length > 2 ? _inputShape[2] : 224;
      int height = _inputShape.length > 2 ? _inputShape[1] : 224;
      image = img.copyResize(image, width: width, height: height);

      final inputTensor = _interpreter!.getInputTensor(0);
      final double inputScale = inputTensor.params.scale;
      final int inputZeroPoint = inputTensor.params.zeroPoint;

      // Prepare input tensor based on datatype (Float32, Uint8, or Int8)
      Object input;
      if (_inputType == TensorType.uint8) {
        input = List.generate(
          1,
          (i) => List.generate(
            height,
            (j) => List.generate(
              width,
              (k) {
                var pixel = image!.getPixelSafe(j, k);
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
                var pixel = image!.getPixelSafe(j, k);
                // Convert uint8 pixel [0..255] to quantized int8 value
                int quantR = ((pixel.r / 255.0) / inputScale + inputZeroPoint).round().clamp(-128, 127);
                int quantG = ((pixel.g / 255.0) / inputScale + inputZeroPoint).round().clamp(-128, 127);
                int quantB = ((pixel.b / 255.0) / inputScale + inputZeroPoint).round().clamp(-128, 127);
                return [quantR, quantG, quantB];
              },
            ),
          ),
        );
      } else {
        // Default FLOAT32 normalization [0.0, 1.0]
        input = List.generate(
          1,
          (i) => List.generate(
            height,
            (j) => List.generate(
              width,
              (k) {
                var pixel = image!.getPixelSafe(j, k);
                return [
                  pixel.r.toDouble() / 255.0,
                  pixel.g.toDouble() / 255.0,
                  pixel.b.toDouble() / 255.0,
                ];
              },
            ),
          ),
        );
      }

      // Prepare output array buffer
      Object output;
      int numClasses = _outputShape.last;
      if (_outputType == TensorType.uint8 || _outputType == TensorType.int8) {
        output = List.generate(1, (_) => List.filled(numClasses, 0));
      } else {
        output = List.generate(1, (_) => List.filled(numClasses, 0.0));
      }

      // Run inference
      _interpreter!.run(input, output);

      stopwatch.stop();
      final inferenceTimeMs = stopwatch.elapsedMilliseconds;
      final memoryMb = ProcessInfo.currentRss / (1024 * 1024);

      // Extract and dequantize probabilities
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

      if (predictions.isEmpty) {
        throw Exception('Predictions list returned empty from TFLite');
      }

      // Find top class index and score
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