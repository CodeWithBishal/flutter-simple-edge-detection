import 'dart:io';
import 'dart:convert';
import 'package:ffi/ffi.dart';
import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

class RfSpot {
  final int x;
  final int y;
  final double rfValue;

  RfSpot({required this.x, required this.y, required this.rfValue});

  factory RfSpot.fromJson(Map<String, dynamic> json) {
    return RfSpot(
      x: json['x'] as int,
      y: json['y'] as int,
      rfValue: json['rf_value'] as double,
    );
  }
}

class TlcCalc {
  static final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libnative_edge_detection.so")
      : DynamicLibrary.process();

  static Future<Map<String, dynamic>> calculateTLC(
    File imagePathToCalc,
    int baseLine,
    int topLine,
  ) async {
    final imagePath = imagePathToCalc.path.toNativeUtf8();

    // Define the new FFI function that returns RF values
    final imageFfiWithRf = dylib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Utf8>, Int32, Int32),
        Pointer<Utf8> Function(Pointer<Utf8>, int, int)>('detect_contour_tlc');

    // Call the function and get the result
    final resultPointer = imageFfiWithRf(imagePath, baseLine, topLine);
    final jsonString = resultPointer.toDartString();

    // Free the allocated memory
    calloc.free(resultPointer);
    calloc.free(imagePath);

    // Parse the JSON string into a list of RfSpot objects
    List<RfSpot> spots = [];
    try {
      final List<dynamic> jsonList = json.decode(jsonString);
      spots = jsonList
          .map((item) => RfSpot.fromJson(item as Map<String, dynamic>))
          .toList();
    } catch (e) {
      print('Error parsing RF values: $e');
    }

    // Save the processed image
    File file = await saveImage(imagePathToCalc.path);

    // Return both the file path and the RF values
    print(spots[0].rfValue);
    return {
      'filePath': file.path,
      'spots': spots,
    };
  }

  // Keep the original method for backward compatibility
  // static Future<String> calculateTLCOriginal(
  //   File imagePathtoCalc,
  //   int baseLine,
  //   int topLine,
  // ) async {
  //   final imagePath = imagePathtoCalc.path.toNativeUtf8();
  //   final imageFfi = dylib.lookupFunction<
  //       Int32 Function(Pointer<Utf8>, Int32, Int32),
  //       int Function(Pointer<Utf8>, int, int)>('detect_contour_tlc');

  //   final int result = imageFfi(imagePath, baseLine, topLine);

  //   print('Result: $result');

  //   calloc.free(imagePath);

  //   if (result != 0) {
  //     File file = await saveImage(imagePathtoCalc.path);
  //     return file.path;
  //   } else {
  //     return imagePathtoCalc.path;
  //   }
  // }

  static Future<File> saveImage(String filePath) async {
    final directory = await getApplicationDocumentsDirectory();
    final fileName = 'processed_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final savedImagePath = '${directory.path}/$fileName';

    final File imageFile = File(filePath);
    await imageFile.copy(savedImagePath);

    return File(savedImagePath);
  }
}
