#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cv;

struct Spot {
    int x;
    int y;
    double rf_value;
};

Mat crop_image(const Mat& image, double left_pct, double right_pct, double top_pct, double bottom_pct) {
    int height = image.rows;
    int width = image.cols;
    
    int left = static_cast<int>(width * left_pct);
    int right = static_cast<int>(width * (1 - right_pct));
    int top = static_cast<int>(height * top_pct);
    int bottom = static_cast<int>(height * (1 - bottom_pct));
    
    Rect roi(left, top, right - left, bottom - top);
    return image(roi);
}

std::pair<Mat, Mat> load_and_preprocess_image(const Mat& img) {
    Mat cropped_img = crop_image(img, 0.05, 0.05, 0.05, 0.05);
    Mat resized_img;
    resize(cropped_img, resized_img, Size(256, 500));
    
    Mat gray_image;
    cvtColor(resized_img, gray_image, COLOR_BGR2GRAY);
    
    Mat blurred_image;
    GaussianBlur(gray_image, blurred_image, Size(5, 5), 0);
    
    return {resized_img, blurred_image};
}

Mat compute_gradients(const Mat& blurred_image) {
    Mat gradient_x, gradient_y;
    Scharr(blurred_image, gradient_x, CV_64F, 1, 0);
    Scharr(blurred_image, gradient_y, CV_64F, 0, 1);
    
    Mat gradient_magnitude;
    magnitude(gradient_x, gradient_y, gradient_magnitude);
    return gradient_magnitude;
}

std::vector<Rect> find_contours(const Mat& gradient_magnitude, double threshold, double min_area_threshold) {
    Mat high_contrast_areas;
    cv::threshold(gradient_magnitude, high_contrast_areas, threshold, 255, THRESH_BINARY);
    high_contrast_areas.convertTo(high_contrast_areas, CV_8U);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(high_contrast_areas, high_contrast_areas, MORPH_CLOSE, kernel);
    
    std::vector<std::vector<Point>> contours;
    std::vector<Rect> rectangles;
    
    findContours(high_contrast_areas, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > min_area_threshold) {
            rectangles.push_back(boundingRect(contour));
        }
    }
    return rectangles;
}

std::vector<Rect> improved_nms(std::vector<Rect>& rectangles, double overlap_thresh) {
    std::vector<Rect> picked;
    if (rectangles.empty()) return picked;
    
    std::vector<double> areas;
    for (const auto& rect : rectangles) {
        areas.push_back(rect.area());
    }
    
    std::vector<size_t> indices(rectangles.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&areas](size_t i1, size_t i2) { return areas[i1] > areas[i2]; });
    
    while (!indices.empty()) {
        size_t current_idx = indices[0];
        picked.push_back(rectangles[current_idx]);
        
        std::vector<size_t> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            Rect& rect1 = rectangles[current_idx];
            Rect& rect2 = rectangles[indices[i]];
            
            Rect intersection = rect1 & rect2;
            double overlap = intersection.area() / static_cast<double>(std::min(rect1.area(), rect2.area()));
            
            if (overlap <= overlap_thresh) {
                new_indices.push_back(indices[i]);
            }
        }
        indices = new_indices;
    }
    
    return picked;
}

std::pair<Mat, std::vector<Spot>> draw_results(const Mat& image, const std::vector<Rect>& rectangles, 
                                              double min_required_area, double max_aspect_ratio) {
    Mat result_img = image.clone();
    std::vector<Spot> spots;
    
    for (const auto& rect : rectangles) {
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;
        double area = rect.area();
        
        if (aspect_ratio <= max_aspect_ratio && area > min_required_area) {
            rectangle(result_img, rect, Scalar(0, 255, 0), 2);
            
            int center_x = rect.x + rect.width / 2;
            int center_y = rect.y + rect.height / 2;
            
            circle(result_img, Point(center_x, center_y), 2, Scalar(0, 0, 255), -1);
            
            double rf_value = 1.0 - static_cast<double>(center_y) / 500;
            spots.push_back({center_x, center_y, rf_value});
            
            std::string text = format("%.3f", rf_value);
            int font_face = FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.4;
            int thickness = 1;
            
            int baseline = 0;
            Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
            
            Point text_org(center_x - text_size.width / 2, center_y + 15);
            putText(result_img, text, text_org, font_face, font_scale, Scalar(255, 0, 0), thickness);
        }
    }
    
    return {result_img, spots};
}

bool detect_contour_tlc(char *image_path) {
    Mat img = imread(image_path);
    if (img.empty()) {
        return false;
    }

    //Check for black color
    Mat hsvImage;
    cvtColor(img, hsvImage, COLOR_BGR2HSV);
    Mat mask;
    inRange(hsvImage, Scalar(0, 0, 0), Scalar(180, 255, 50), mask); // Threshold for black color
    if (countNonZero(mask) > 0) {
        return false; // Return false if black color is found
    }
    
    std::pair<Mat, Mat> preprocess_result = load_and_preprocess_image(img);
    Mat gradient_magnitude = compute_gradients(blurred_image);
    
    double initial_threshold = 50;
    double initial_min_area_threshold = 200;
    double min_required_area = 250;
    double max_aspect_ratio = 3;
    
    std::vector<Rect> rectangles = find_contours(gradient_magnitude, initial_threshold, initial_min_area_threshold);
    rectangles = improved_nms(rectangles, 0.3);
    
    std::pair<Mat, std::vector<Spot>> results = draw_results(resized_img, rectangles, min_required_area, max_aspect_ratio);
    
    imwrite(image_path, result_img);
    return true;
}
