import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:simple_edge_detection/edge_detection.dart';
import 'package:flutter/material.dart';
import 'package:simple_edge_detection/edge_detection_shape.dart';

class ImagePreview extends StatefulWidget {
  const ImagePreview(
      {super.key, required this.imagePath, required this.edgeDetectionResult});

  final String imagePath;
  final EdgeDetectionResult edgeDetectionResult;

  @override
  ImagePreviewState createState() => ImagePreviewState();
}

class ImagePreviewState extends State<ImagePreview> {
  GlobalKey imageWidgetKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          Center(child: Text('Loading ...')),
          Image.file(File(widget.imagePath),
              fit: BoxFit.contain, key: imageWidgetKey),
          FutureBuilder<ui.Image>(
              future: loadUiImage(widget.imagePath),
              builder:
                  (BuildContext context, AsyncSnapshot<ui.Image> snapshot) {
                return _getEdgePaint(snapshot, context);
              }),
        ],
      ),
    );
  }

  Widget _getEdgePaint(
      AsyncSnapshot<ui.Image> imageSnapshot, BuildContext context) {
    if (imageSnapshot.connectionState == ConnectionState.waiting) {
      return Container();
    }

    if (imageSnapshot.hasError) {
      return Text('Error: ${imageSnapshot.error}');
    }

    // ignore: unnecessary_null_comparison
    if (widget.edgeDetectionResult == null) {
      return Container();
    }

    final keyContext = imageWidgetKey.currentContext;

    if (keyContext == null) {
      return Container();
    }

    final box = keyContext.findRenderObject() as RenderBox;

    return EdgeDetectionShape(
      originalImageSize: Size((imageSnapshot.data?.width ?? 0).toDouble(),
          (imageSnapshot.data?.height ?? 0).toDouble()),
      renderedImageSize: Size(box.size.width, box.size.height),
      edgeDetectionResult: widget.edgeDetectionResult,
    );
  }

  Future<ui.Image> loadUiImage(String imageAssetPath) async {
    final Uint8List data = await File(imageAssetPath).readAsBytes();
    final Completer<ui.Image> completer = Completer();
    ui.decodeImageFromList(Uint8List.view(data.buffer), (ui.Image image) {
      return completer.complete(image);
    });
    return completer.future;
  }
}
