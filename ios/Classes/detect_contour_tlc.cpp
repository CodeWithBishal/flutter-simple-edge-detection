#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;

Mat crop_image(const Mat& image, double left_pct, double right_pct, double top_pct, double bottom_pct) {
    int height = image.rows;
    int width = image.cols;
    int left = static_cast<int>(width * left_pct);
    int right = static_cast<int>(width * right_pct);
    int top = static_cast<int>(height * top_pct);
    int bottom = static_cast<int>(height * bottom_pct);
    Rect cropped_region(left, top, width - left - right, height - top - bottom);
    return image(cropped_region);
}

pair<Mat, Mat> load_and_preprocess_image(const Mat& img) {
    Mat cropped_img = crop_image(img, 0.10, 0.10, 0.05, 0.05);
    Mat resized_img;
    resize(cropped_img, resized_img, Size(256, 500));
    Mat gray_image;
    cvtColor(resized_img, gray_image, COLOR_BGR2GRAY);
    Mat blurred_image;
    GaussianBlur(gray_image, blurred_image, Size(5, 5), 0);
    return make_pair(resized_img, blurred_image);
}

Mat compute_gradients(const Mat& blurred_image) {
    Mat gradient_x, gradient_y;
    Scharr(blurred_image, gradient_x, CV_64F, 1, 0);
    Scharr(blurred_image, gradient_y, CV_64F, 0, 1);
    Mat gradient_magnitude;
    magnitude(gradient_x, gradient_y, gradient_magnitude);
    return gradient_magnitude;
}

vector<Rect> find_contours(const Mat& gradient_magnitude, double thresh_value, double min_area_threshold) {
    Mat high_contrast_areas;
    threshold(gradient_magnitude, high_contrast_areas, thresh_value, 255, THRESH_BINARY);
    high_contrast_areas.convertTo(high_contrast_areas, CV_8U);
    vector<vector<Point>> contours;
    findContours(high_contrast_areas, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect> rectangles;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > min_area_threshold) {
            Rect rect = boundingRect(contour);
            rectangles.push_back(rect);
        }
    }
    return rectangles;
}

vector<Rect> non_max_suppression(const vector<Rect>& rectangles, double overlap_thresh) {
    if (rectangles.empty()) return {};

    vector<int> pick;
    vector<int> idxs(rectangles.size());
    iota(idxs.begin(), idxs.end(), 0);

    sort(idxs.begin(), idxs.end(), [&rectangles](int i1, int i2) {
        return rectangles[i1].br().y < rectangles[i2].br().y;
    });

    while (!idxs.empty()) {
        int last = idxs.size() - 1;
        int i = idxs[last];
        pick.push_back(i);
        vector<int> suppress = {last};

        for (int pos = 0; pos < last; ++pos) {
            int j = idxs[pos];
            int xx1 = max(rectangles[i].x, rectangles[j].x);
            int yy1 = max(rectangles[i].y, rectangles[j].y);
            int xx2 = min(rectangles[i].br().x, rectangles[j].br().x);
            int yy2 = min(rectangles[i].br().y, rectangles[j].br().y);
            int w = max(0, xx2 - xx1 + 1);
            int h = max(0, yy2 - yy1 + 1);
            double overlap = static_cast<double>(w * h) / (rectangles[j].area());

            if (overlap > overlap_thresh) {
                suppress.push_back(pos);
            }
        }

        for (int pos = suppress.size() - 1; pos >= 0; --pos) {
            idxs.erase(idxs.begin() + suppress[pos]);
        }
    }

    vector<Rect> result;
    for (int i : pick) {
        result.push_back(rectangles[i]);
    }
    return result;
}

void draw_text(Mat& image, int x, int y, int x2, int y2, double y_normalized) {
    string text = format("%.2f", y_normalized);
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 1;
    int baseline = 0;
    Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
    Point text_org((x + x2 - text_size.width) / 2, y - text_size.height - 5);
    putText(image, text, text_org, font_face, font_scale, Scalar(255, 0, 0), thickness);
}

void draw_rectangles(Mat& image, const vector<Rect>& rectangles, double min_required_area, double max_aspect_ratio) {
    for (const auto& rect : rectangles) {
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;
        double area = rect.area();
        if (aspect_ratio <= max_aspect_ratio && area > min_required_area) {
            rectangle(image, rect, Scalar(0, 255, 0), 2);
            Point center((rect.x + rect.br().x) / 2, (rect.y + rect.br().y) / 2);
            circle(image, center, 1, Scalar(0, 0, 255), -1);
            double y_normalized = 1.0 - static_cast<double>(center.y) / 500;
            draw_text(image, rect.x, rect.y, rect.br().x, rect.br().y, y_normalized);
        }
    }
}
extern "C" __attribute__((visibility("default"))) __attribute__((used))
bool detect_contour_tlcc(char *path) {
    Mat img = imread(path);
    cout << "Processing image at path: " << path << endl;
    if (img.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return false;
    }

    try {
        pair<Mat, Mat> result = load_and_preprocess_image(img);
        Mat resized_img = result.first;
        Mat blurred_image = result.second;
        
        Mat gradient_magnitude = compute_gradients(blurred_image);
        double initial_threshold = 50;
        double initial_min_area_threshold = 200;
        int min_required_rectangles = 7;
        double min_required_area = 250;
        double max_aspect_ratio = 3;

        int num_rectangles = 0;
        while (num_rectangles < min_required_rectangles) {
            vector<Rect> rectangles = find_contours(gradient_magnitude, initial_threshold, initial_min_area_threshold);
            rectangles = non_max_suppression(rectangles, 0.2);
            draw_rectangles(resized_img, rectangles, min_required_area, max_aspect_ratio);
            num_rectangles = rectangles.size();
            initial_min_area_threshold -= 100;
        }

        return true;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return false;
    }
}