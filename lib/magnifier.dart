import 'dart:ui';
import 'package:flutter/material.dart';

import 'magnifier_painter.dart';

class Magnifier extends StatefulWidget {
  const Magnifier(
      {super.key,
      required this.child,
      required this.position,
      this.visible = true,
      this.scale = 1.5,
      this.size = const Size(160, 160)});

  final Widget child;
  final Offset position;
  final bool visible;
  final double scale;
  final Size size;

  @override
  MagnifierState createState() => MagnifierState();
}

class MagnifierState extends State<Magnifier> {
  late Size _magnifierSize;
  late double _scale;
  late Matrix4 _matrix;

  @override
  void initState() {
    _magnifierSize = widget.size;
    _scale = widget.scale;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _calculateMatrix();
    });

    super.initState();
  }

  @override
  void didUpdateWidget(Magnifier oldWidget) {
    super.didUpdateWidget(oldWidget);

    _calculateMatrix();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [widget.child, if (widget.visible) _getMagnifier(context)],
    );
  }

  void _calculateMatrix() {
    setState(() {
      double newX = widget.position.dx - (_magnifierSize.width / 2 / _scale);
      double newY = widget.position.dy - (_magnifierSize.height / 2 / _scale);

      if (_bubbleCrossesMagnifier()) {
        final box = context.findRenderObject() as RenderBox;
        newX -= ((box.size.width - _magnifierSize.width) / _scale);
      }

      final Matrix4 updatedMatrix = Matrix4.identity()
        ..scale(_scale, _scale)
        ..translate(-newX, -newY);

      _matrix = updatedMatrix;
    });
  }

  Widget _getMagnifier(BuildContext context) {
    return Align(
      alignment: _getAlignment(),
      child: ClipOval(
        child: BackdropFilter(
          filter: ImageFilter.matrix(_matrix.storage),
          child: CustomPaint(
            painter: MagnifierPainter(
                color: Theme.of(context).colorScheme.secondary),
            size: _magnifierSize,
          ),
        ),
      ),
    );
  }

  Alignment _getAlignment() {
    if (_bubbleCrossesMagnifier()) {
      return Alignment.topRight;
    }

    return Alignment.topLeft;
  }

  bool _bubbleCrossesMagnifier() =>
      widget.position.dx < widget.size.width &&
      widget.position.dy < widget.size.height;
}
