import 'dart:io';
import 'dart:convert';
import 'package:ffi/ffi.dart';
import 'dart:ffi';
import 'package:flutter/foundation.dart';
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

  // ── Original method (unchanged) ─────────────────────────────────────────────
  static Future<Map<String, dynamic>> calculateTLC(
    File imagePathToCalc,
    int baseLine,
    int topLine,
  ) async {
    final imagePath = imagePathToCalc.path.toNativeUtf8();

    final imageFfiWithRf = dylib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Utf8>, Int32, Int32),
        Pointer<Utf8> Function(Pointer<Utf8>, int, int)>('detect_contour_tlc');

    final resultPointer = imageFfiWithRf(imagePath, baseLine, topLine);
    final jsonString = resultPointer.toDartString();

    calloc.free(resultPointer);
    calloc.free(imagePath);

    List<RfSpot> spots = [];
    try {
      final List<dynamic> jsonList = json.decode(jsonString);
      spots = jsonList
          .map((item) => RfSpot.fromJson(item as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('Error parsing RF values: $e');
    }

    File file = await saveImage(imagePathToCalc.path);

    debugPrint(spots.isEmpty ? 'No spots detected' : 'First spot rf: ${spots[0].rfValue}');
    return {
      'filePath': file.path,
      'spots': spots,
    };
  }

  // ── NEW: calculateTLCWithHints ───────────────────────────────────────────────
  /// Same as [calculateTLC] but also accepts a list of manually annotated
  /// bounding boxes in original image pixel coordinates.
  ///
  /// Each box is `{'x1': int, 'y1': int, 'x2': int, 'y2': int}`.
  /// These are serialised to JSON and forwarded to the C++ function
  /// `detect_contour_tlc_with_hints`, which merges them with auto-detected
  /// spots before computing Rf values.
  static Future<Map<String, dynamic>> calculateTLCWithHints(
    File imagePathToCalc,
    int baseLine,
    int topLine,
    List<Map<String, int>> manualBoxes,
  ) async {
    // If no manual boxes provided, fall back to the standard call
    if (manualBoxes.isEmpty) {
      return calculateTLC(imagePathToCalc, baseLine, topLine);
    }

    final imagePath = imagePathToCalc.path.toNativeUtf8();
    final boxesJson = json.encode(manualBoxes).toNativeUtf8();

    final ffiFunc = dylib.lookupFunction<
        Pointer<Utf8> Function(Pointer<Utf8>, Int32, Int32, Pointer<Utf8>),
        Pointer<Utf8> Function(
            Pointer<Utf8>, int, int, Pointer<Utf8>)>('detect_contour_tlc_with_hints');

    final resultPointer = ffiFunc(imagePath, baseLine, topLine, boxesJson);
    final jsonString = resultPointer.toDartString();

    calloc.free(resultPointer);
    calloc.free(imagePath);
    calloc.free(boxesJson);

    List<RfSpot> spots = [];
    try {
      final List<dynamic> jsonList = json.decode(jsonString);
      spots = jsonList
          .map((item) => RfSpot.fromJson(item as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('Error parsing RF values (with hints): $e');
    }

    File file = await saveImage(imagePathToCalc.path);

    return {
      'filePath': file.path,
      'spots': spots,
    };
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
