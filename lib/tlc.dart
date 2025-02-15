import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

class TlcCalc {
  static final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libnative_edge_detection.so")
      : DynamicLibrary.process();
  static Future<String> calculateTLC(
    File imagePathtoCalc,
    int baseLine,
    int topLine,
  ) async {
    final imagePath = imagePathtoCalc.path.toNativeUtf8();
    final imageFfi = dylib.lookupFunction<
        Int32 Function(Pointer<Utf8>, Int32, Int32),
        int Function(Pointer<Utf8>, int, int)>('detect_contour_tlc');

    final int result = imageFfi(imagePath, baseLine, topLine);

    calloc.free(imagePath);

    print(result);

    if (result != 0) {
      File file = await saveImage(imagePath.toDartString());
      return file.path;
    } else {
      return imagePathtoCalc.path;
    }

//ffi
  }

  static Future<File> saveImage(String filePath) async {
    final directory = await getApplicationDocumentsDirectory();
    final fileName = 'processed_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final savedImagePath = '${directory.path}/$fileName';

    final File imageFile = File(filePath);
    await imageFile.copy(savedImagePath);

    return File(savedImagePath);
  }
}
