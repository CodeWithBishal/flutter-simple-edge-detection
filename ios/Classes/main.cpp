extern "C" {
    __attribute__((visibility("default"))) __attribute__((used))
    int detect_contour_tlc(char *, int baseline_y, int topline_y);
}
#include "detect_contour_tlc.cpp"