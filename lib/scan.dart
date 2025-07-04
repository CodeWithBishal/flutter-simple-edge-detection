import 'dart:async';
import 'dart:io';
import 'package:simple_edge_detection/cropping_preview.dart';
import 'package:simple_edge_detection/edge_detection.dart';
import 'package:flutter/material.dart';
import 'package:simple_edge_detection/image_view.dart';
import 'package:path_provider/path_provider.dart';
import 'package:simple_edge_detection/edge_detector.dart';

class Scan extends StatefulWidget {
  final String imagePath;
  const Scan({
    super.key,
    required this.imagePath,
  });

  @override
  ScanState createState() => ScanState();
}

class ScanState extends State<Scan> {
  String? croppedImagePath;
  EdgeDetectionResult? edgeDetectionResult;
  late String? imagePath;
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    enter(widget.imagePath);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Crop Image'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            if (!isLoading) {
              Navigator.pop(context);
            }
          },
        ),
        actions: [
          imagePath != null
              ? TextButton(
                  onPressed: () {
                    if (croppedImagePath == null) {
                      _processImage(imagePath!, edgeDetectionResult!);
                    }

                    setState(() {
                      imagePath = null;
                      edgeDetectionResult = null;
                      croppedImagePath = null;
                    });
                  },
                  child: Text(
                    "Crop &  Process",
                  ),
                )
              : Container(),
        ],
      ),
      body: Stack(
        children: <Widget>[
          _getMainWidget(),
          if (isLoading)
            Center(
              child: CircularProgressIndicator(),
            ),
          _getBottomBar(),
        ],
      ),
    );
  }

  Widget _getMainWidget() {
    if (croppedImagePath != null) {
      return ImageView(imagePath: croppedImagePath!);
    }

    if (edgeDetectionResult != null && imagePath != null) {
      return ImagePreview(
        imagePath: imagePath!,
        edgeDetectionResult: edgeDetectionResult!,
      );
    } else {
      return Container();
    }
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future _detectEdges(String filePath) async {
    if (!mounted) {
      return;
    }

    setState(() {
      imagePath = filePath;
    });

    try {
      EdgeDetectionResult result = await EdgeDetector().detectEdges(filePath);
      if (result.bottomLeft.dx == 0) {
        result = EdgeDetectionResult(
          topLeft: Offset(0.25, 0.25),
          topRight: Offset(0.75, 0.25),
          bottomLeft: Offset(0.25, 0.75),
          bottomRight: Offset(0.75, 0.75),
        );
      }
      setState(() {
        edgeDetectionResult = result;
      });
    } catch (e) {
      Future.microtask(() {
        if (!mounted) return;
        Navigator.pop(context, widget.imagePath);
      });
    }
  }

  Future _processImage(
      String filePath, EdgeDetectionResult edgeDetectionResult) async {
    if (!mounted) {
      return;
    }

    setState(() {
      isLoading = true;
    });

    try {
      bool result =
          await EdgeDetector().processImage(filePath, edgeDetectionResult);

      if (result == false) {
        setState(() {
          isLoading = false;
        });
        return;
      }

      await _saveProcessedImage(filePath);
      // String savedImagePath = await TlcCalc.calculateTLC(File(preCalc));
      String savedImagePath = "";
      setState(() {
        imageCache.clearLiveImages();
        imageCache.clear();
        croppedImagePath = savedImagePath;
        isLoading = false;
      });

      Future.microtask(() {
        if (!mounted) return;
        Navigator.pop(context, savedImagePath);
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      if (!mounted) return;
      Navigator.pop(context, widget.imagePath);
    }
  }

  Future<String> _saveProcessedImage(String filePath) async {
    final directory = await getApplicationDocumentsDirectory();
    final fileName = 'processed_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final savedImagePath = '${directory.path}/$fileName';

    final File imageFile = File(filePath);
    await imageFile.copy(savedImagePath);
    return savedImagePath;
  }

  void enter(String filePath) async {
    setState(() {
      imagePath = filePath;
    });
    _detectEdges(filePath);
  }

  Padding _getBottomBar() {
    return Padding(
        padding: EdgeInsets.only(bottom: 32),
        child: Align(alignment: Alignment.bottomCenter, child: Container()));
  }
}
