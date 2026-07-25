#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct Spot
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int cls;
};

// Runs YOLOv8 ONNX detection against an already-loaded image (or lane crop).
// Used for both the spot model and the lane/strip model — construct one
// instance per model.
//
// Session lifetime: each instance owns its own Ort::Session, but *not* its
// own Ort::Env — it binds to a single process-global Ort::Env (see
// SpotDetector.cpp). ONNX Runtime requires exactly one Env per process;
// creating/destroying multiple Envs corrupts the arena allocator on Android
// ARM64 (MTE-enabled devices), causing SIGSEGV in subsequent std::vector
// reallocations. A SpotDetector instance is safe to construct fresh within
// a single top-level call (e.g. once per lane, or once for the whole call)
// and call detect() on multiple times — just don't cache one as a
// long-lived singleton across separate top-level FFI calls.
class SpotDetector
{
public:
    // modelPath is a plain (UTF-8/ASCII) path; the Windows-vs-POSIX wide
    // string distinction ONNX Runtime requires is handled internally in
    // the .cpp, so callers never need to deal with wstring conversion or
    // #ifdefs themselves.
    explicit SpotDetector(const std::string& modelPath);
    std::vector<Spot> detect(const cv::Mat& image, float confThreshold = 0.0009f, float iouThreshold = 0.45f);

private:
    Ort::Session session;
};
