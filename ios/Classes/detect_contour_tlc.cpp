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
    
    return std::make_pair(resized_img, blurred_image);
}

Mat compute_gradients(const Mat& blurred_image) {
    Mat gradient_x, gradient_y;
    Scharr(blurred_image, gradient_x, CV_64F, 1, 0);
    Scharr(blurred_image, gradient_y, CV_64F, 0, 1);
    
    Mat gradient_magnitude;
    magnitude(gradient_x, gradient_y, gradient_magnitude);
    return gradient_magnitude;
}

std::vector<Rect> find_contours(const Mat& gradient_magnitude, double threshold, 
                               double min_area_threshold, int baseline_y = -1, int topline_y = -1) {
    Mat high_contrast_areas;
    cv::threshold(gradient_magnitude, high_contrast_areas, threshold, 255, THRESH_BINARY);
    high_contrast_areas.convertTo(high_contrast_areas, CV_8U);
    
    // If baseline and topline are provided, mask out areas outside the region
    if (baseline_y != -1 && topline_y != -1) {
        // Create a mask for the region between baseline and topline
        Mat mask = Mat::zeros(high_contrast_areas.size(), CV_8U);
        rectangle(mask, Point(0, topline_y), Point(mask.cols - 1, baseline_y), Scalar(255), -1);
        high_contrast_areas = high_contrast_areas & mask;
    }
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(high_contrast_areas, high_contrast_areas, MORPH_CLOSE, kernel);
    
    std::vector<std::vector<Point>> contours;
    std::vector<Rect> rectangles;
    
    findContours(high_contrast_areas, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > min_area_threshold) {
            Rect rect = boundingRect(contour);
            
            // Only include rectangles that are entirely within the baseline-topline region
            if (baseline_y == -1 || topline_y == -1 || 
                (rect.y >= topline_y && rect.y + rect.height <= baseline_y)) {
                rectangles.push_back(rect);
            }
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
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
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
                                              double min_required_area, double max_aspect_ratio,
                                              int baseline_y = -1, int topline_y = -1) {
    Mat result_img = image.clone();
    std::vector<Spot> spots;
    
    // Calculate the range for Rf value
    double bottom_position = (baseline_y != -1) ? baseline_y : image.rows;
    double top_position = (topline_y != -1) ? topline_y : 0;
    double total_distance = bottom_position - top_position;
    
    for (const auto& rect : rectangles) {
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;
        double area = rect.area();
        
        if (aspect_ratio <= max_aspect_ratio && area > min_required_area) {
            int center_x = rect.x + rect.width / 2;
            int center_y = rect.y + rect.height / 2;
            
            // Only process spots between baseline and topline when provided
            if (baseline_y == -1 || topline_y == -1 || 
                (center_y >= topline_y && center_y <= baseline_y)) {
                
                rectangle(result_img, rect, Scalar(0, 255, 0), 2);
                circle(result_img, Point(center_x, center_y), 2, Scalar(0, 0, 255), -1);
                
                // Calculate Rf value based on provided lines
                double rf_value;
                if (baseline_y != -1 && topline_y != -1) {
                    rf_value = (bottom_position - center_y) / total_distance;
                } else {
                    rf_value = 1.0 - static_cast<double>(center_y) / 500;
                }
                
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
    }
    
    return std::make_pair(result_img, spots);
}

const char* detect_contour_tlc(char *image_path, int baseline_y, int topline_y) {
    Mat img = imread(image_path);
    if (img.empty()) {
        return strdup("[]");
    }
    
    // Get preprocessing results
    std::pair<Mat, Mat> preprocess_result = load_and_preprocess_image(img);
    Mat resized_img = preprocess_result.first;
    Mat blurred_image = preprocess_result.second;
    
    // Calculate scaling factors for both dimensions
    double scale_y = static_cast<double>(resized_img.rows) / img.rows;
    double scale_x = static_cast<double>(resized_img.cols) / img.cols;
    
    // Scale the baseline and topline positions according to the resized image
    double crop_compensation = 0.05;
    int scaled_baseline_y = -1;
    int scaled_topline_y = -1;
    
    if (baseline_y != -1) {
        double compensated_y = (baseline_y / (1.0 - 2 * crop_compensation)) - (img.rows * crop_compensation);
        scaled_baseline_y = static_cast<int>(compensated_y * scale_y);
    }
    
    if (topline_y != -1) {
        double compensated_y = (topline_y / (1.0 - 2 * crop_compensation)) - (img.rows * crop_compensation);
        scaled_topline_y = static_cast<int>(compensated_y * scale_y);
    }
    
    // Ensure the scaled positions are within bounds
    if (scaled_baseline_y != -1) {
        scaled_baseline_y = std::min(std::max(scaled_baseline_y, 0), resized_img.rows - 1);
    }
    if (scaled_topline_y != -1) {
        scaled_topline_y = std::min(std::max(scaled_topline_y, 0), resized_img.rows - 1);
    }
    
    Mat gradient_magnitude = compute_gradients(blurred_image);
    
    double initial_threshold = 40;  // Lowered from 50
    double initial_min_area_threshold = 180;  // Lowered from 200
    double min_required_area = 220;  // Lowered from 250
    double max_aspect_ratio = 2.5;  // Slightly reduced from 3
    
    
    // Draw the lines with proper position compensation
    if (scaled_baseline_y != -1) {
        line(resized_img, Point(0, scaled_baseline_y), 
             Point(resized_img.cols - 1, scaled_baseline_y),
             Scalar(0, 0, 255), 2); // Red line for baseline
    }
    
    if (scaled_topline_y != -1) {
        line(resized_img, Point(0, scaled_topline_y), 
             Point(resized_img.cols - 1, scaled_topline_y),
             Scalar(255, 0, 0), 2); // Blue line for topline
    }
    
    // Pass the scaled line positions to find_contours
    std::vector<Rect> rectangles = find_contours(gradient_magnitude, initial_threshold, 
                                               initial_min_area_threshold,
                                               scaled_baseline_y, scaled_topline_y);
    rectangles = improved_nms(rectangles, 0.3);
    
    // Pass the scaled line positions to draw_results
    std::pair<Mat, std::vector<Spot>> results = draw_results(resized_img, rectangles, 
                                                            min_required_area, max_aspect_ratio,
                                                            scaled_baseline_y, scaled_topline_y);
    Mat result_img = results.first;
    std::vector<Spot> spots = results.second;
    
    imwrite(image_path, result_img);
    
    // Construct JSON string with RF values
    std::string json = "[";
    for (size_t i = 0; i < spots.size(); ++i) {
        json += "{\"x\":" + std::to_string(spots[i].x) + 
            ",\"y\":" + std::to_string(spots[i].y) + 
            ",\"rf_value\":" + format("%.3f", spots[i].rf_value) + "}";
        if (i < spots.size() - 1) json += ",";
    }
    json += "]";
    
    return strdup(json.c_str()); 
}