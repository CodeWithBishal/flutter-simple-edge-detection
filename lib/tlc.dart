import 'dart:io';

import 'package:ffi/ffi.dart';
import 'dart:ffi';

class TlcCalc {
  static final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libOpenCV_ffi.so")
      : DynamicLibrary.process();
  static File calculateTLC(File imagePathtoCalc) {
    final imagePath = imagePathtoCalc.path.toNativeUtf8();
    final imageFfi = dylib.lookupFunction<Bool Function(Pointer<Utf8>),
        bool Function(Pointer<Utf8>)>('detect_contour_tlc');
    if (imageFfi(imagePath)) {
      saveImage(imagePath.toDartString(), imagePathtoCalc.path);
      return File(imagePath.toDartString());
    } else {
      return imagePathtoCalc;
    }

//ffi
  }

  static void saveImage(String finalImage, filePath) async {
    await File(filePath).writeAsBytes(File(finalImage).readAsBytesSync());
  }
}
