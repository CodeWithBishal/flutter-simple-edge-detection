#include <opencv2/opencv.hpp>
#include <numeric>
#include <sstream>
#include <iomanip>
using namespace cv;

bool detect_contour_tlc(char *path) {
    Mat img = imread(path);

    //Check for black color
    Mat hsvImage;
    cvtColor(img, hsvImage, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsvImage, Scalar(0, 0, 0), Scalar(180, 255, 50), mask); // Threshold for black color
    if (countNonZero(mask) > 0) {
        return false; // Return false if black color is found
    }

    resize(img, img, Size(256, 500));
    Mat grayImage, blurredImage;
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);

    Mat gradientX, gradientY, gradientMagnitude;
    Scharr(blurredImage, gradientX, CV_64F, 1, 0);
    Scharr(blurredImage, gradientY, CV_64F, 0, 1);
    magnitude(gradientX, gradientY, gradientMagnitude);

    int initialThreshold = 50;
    int initialMinAreaThreshold = 200;
    int numRectangles = 0;
    int minRequiredRectangles = 7;
    int minRequiredArea = 250;

    std::vector<Point> prevTextPositions;

    while (numRectangles < minRequiredRectangles) {
        int threshold = initialThreshold;
        int minAreaThreshold = initialMinAreaThreshold;
        Mat highContrastAreas = gradientMagnitude > threshold;
        std::vector<std::vector<Point>> contours;
        findContours(highContrastAreas.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::vector<Rect> rectangles;
        for (auto& contour : contours) {
            double area = contourArea(contour);
            if (area > minAreaThreshold) {
                Rect rect = boundingRect(contour);
                rectangles.push_back(rect);
            }
        }

        // Non-maximum suppression
        std::vector<int> pick;
        std::vector<int> x1, y1, x2, y2, area;
        for (const auto& rect : rectangles) {
            x1.push_back(rect.x);
            y1.push_back(rect.y);
            x2.push_back(rect.x + rect.width);
            y2.push_back(rect.y + rect.height);
            area.push_back(rect.width * rect.height);
        }

        std::vector<int> idxs(area.size());
        std::iota(idxs.begin(), idxs.end(), 0);

        sort(idxs.begin(), idxs.end(), [&](int i, int j) {
            return y2[i] > y2[j];
        });

        while (!idxs.empty()) {
            int last = idxs.size() - 1;
            int i = idxs[last];
            pick.push_back(i);
            std::vector<int> suppress = {last};
            for (int pos = 0; pos < last; pos++) {
                int j = idxs[pos];
                int xx1 = max(x1[i], x1[j]);
                int yy1 = max(y1[i], y1[j]);
                int xx2 = min(x2[i], x2[j]);
                int yy2 = min(y2[i], y2[j]);
                int w = max(0, xx2 - xx1 + 1);
                int h = max(0, yy2 - yy1 + 1);
                double overlap = static_cast<double>(w * h) / area[j];

                if (overlap > 0.2) {
                    suppress.push_back(pos);
                }
            }
            idxs.erase(idxs.begin() + suppress.back());
            suppress.pop_back();
        }

        for (int i : pick) {
            int x = rectangles[i].x;
            int y = rectangles[i].y;
            int x2 = rectangles[i].x + rectangles[i].width;
            int y2 = rectangles[i].y + rectangles[i].height;
            int centerX = (x + x2) / 2;
            int centerY = (y + y2) / 2;

            bool overlapDetected = any_of(prevTextPositions.begin(), prevTextPositions.end(), [&](const Point& prev) {
                return abs(prev.x - centerX) < 5 && abs(prev.y - centerY) < 5;
            });

            if (!overlapDetected) {
                circle(img, Point(centerX, centerY), 5, Scalar(0, 0, 255), -1);

                double yNormalized = 1.0 - static_cast<double>(centerY) / 500;

                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << yNormalized;
                String text = ss.str();

                int baseline = 0;
                Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                Point textOrigin(x + (rectangles[i].width - textSize.width) / 2, y - baseline);

                overlapDetected = any_of(prevTextPositions.begin(), prevTextPositions.end(), [&](const Point& prev) {
                    return abs(prev.x - textOrigin.x) < textSize.width && abs(prev.y - textOrigin.y) < textSize.height;
                });

                if (!overlapDetected) {
                    putText(img, text, textOrigin, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));
                    prevTextPositions.push_back(textOrigin);
                }
            }
        }
        numRectangles = pick.size();
        initialMinAreaThreshold -= 10;
    }

    imwrite(path, img);
    return true;
}