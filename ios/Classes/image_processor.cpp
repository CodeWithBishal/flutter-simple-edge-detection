#include "image_processor.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

Point2f computePoint(int p1, int p2) {
    Point2f pt;
    pt.x = p1;
    pt.y = p2;
    return pt;
}

Mat ImageProcessor::process_image(Mat img, float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4) {
    // Remove the conversion to grayscale
    Mat dst = ImageProcessor::crop_and_transform(img, x1, y1, x2, y2, x3, y3, x4, y4);
    return dst; // Return the cropped color image
}

Mat ImageProcessor::crop_and_transform(Mat img, float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4) {
    vector<Point2f> img_pts;
    img_pts.push_back(computePoint(x1, y1));
    img_pts.push_back(computePoint(x2, y2));
    img_pts.push_back(computePoint(x3, y3));
    img_pts.push_back(computePoint(x4, y4));

    // Calculate bounding box
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::min();
    float maxY = std::numeric_limits<float>::min();

    for (const auto& pt : img_pts) {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    float width = maxX - minX;
    float height = maxY - minY;

    // Set destination image size based on the bounding box
    Mat dst = Mat::zeros(height, width, CV_8UC3);

    vector<Point2f> dst_pts;
    dst_pts.push_back(Point(0, 0));
    dst_pts.push_back(Point(width - 1, 0));
    dst_pts.push_back(Point(0, height - 1));
    dst_pts.push_back(Point(width - 1, height - 1));

    Mat transformation_matrix = getPerspectiveTransform(img_pts, dst_pts);
    warpPerspective(img, dst, transformation_matrix, dst.size());

    return dst;
}
