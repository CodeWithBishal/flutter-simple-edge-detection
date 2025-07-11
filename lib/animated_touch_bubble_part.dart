import 'package:flutter/material.dart';

class AnimatedTouchBubblePart extends StatefulWidget {
  const AnimatedTouchBubblePart(
      {super.key, required this.dragging, required this.size});

  final bool dragging;
  final double size;

  @override
  AnimatedTouchBubblePartState createState() => AnimatedTouchBubblePartState();
}

class AnimatedTouchBubblePartState extends State<AnimatedTouchBubblePart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Color?> _colorAnimation;
  late Animation<double> _sizeAnimation;

  @override
  void didChangeDependencies() {
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    _sizeAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(_controller);

    _colorAnimation = ColorTween(
            begin:
                Theme.of(context).colorScheme.secondary.withValues(alpha: 0.5),
            end: Theme.of(context).colorScheme.secondary.withValues(alpha: 0.0))
        .animate(
            CurvedAnimation(parent: _controller, curve: Interval(0.5, 1.0)));

    _controller.repeat();
    super.didChangeDependencies();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Center(
            child: Container(
                width: widget.dragging ? 0 : widget.size / 2,
                height: widget.dragging ? 0 : widget.size / 2,
                decoration: BoxDecoration(
                    color: Theme.of(context)
                        .colorScheme
                        .secondary
                        .withValues(alpha: 0.5),
                    borderRadius: widget.dragging
                        ? BorderRadius.circular(widget.size)
                        : BorderRadius.circular(widget.size / 4)))),
        AnimatedBuilder(
            builder: (BuildContext context, Widget? child) {
              return Center(
                  child: Container(
                      width: widget.dragging
                          ? 0
                          : widget.size * _sizeAnimation.value,
                      height: widget.dragging
                          ? 0
                          : widget.size * _sizeAnimation.value,
                      decoration: BoxDecoration(
                          border: Border.all(
                              color:
                                  _colorAnimation.value ?? Colors.transparent,
                              width: widget.size / 20),
                          borderRadius: widget.dragging
                              ? BorderRadius.zero
                              : BorderRadius.circular(
                                  widget.size * _sizeAnimation.value / 2))));
            },
            animation: _controller)
      ],
    );
  }
}
