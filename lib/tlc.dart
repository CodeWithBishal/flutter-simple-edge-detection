import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:ffi';

import 'package:path_provider/path_provider.dart';

class TlcCalc {
  static final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libnative_edge_detection.so")
      : DynamicLibrary.process();
  static Future<String> calculateTLC(File imagePathtoCalc) async {
    final imagePath = imagePathtoCalc.path.toNativeUtf8();
    final imageFfi = dylib.lookupFunction<Bool Function(Pointer<Utf8>),
        bool Function(Pointer<Utf8>)>('detect_contour_tlc');
    if (imageFfi(imagePath)) {
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
