// ffi_exports.cpp
// -----------------------------------------------------------------------
// C-API FFI wrapper for the New C++ TLC backend.
// Exports:
//   process_tlc(const char* args)      — main analysis entry point (runs
//                                        lane + spot detection from scratch)
//   add_manual_spots(const char* args) — adds user-drawn spots to an
//                                        already-processed image without
//                                        re-running any ONNX model; see its
//                                        own doc comment below
//   free_result(const char* ptr)       — frees the malloc'd result string
//                                        returned by either of the above
//
// process_tlc input format (pipe-delimited string):
//   image_path|model_path|baseline|topline|plot_output_path|manual_spots_str|strip_model_path
//
// manual_spots_str format:
//   x1,y1,x2,y2;x1,y1,x2,y2;...   (semicolon-separated bounding boxes, in
//   *absolute* image-pixel coordinates)
//
// strip_model_path is optional. When empty (or when loading/running it
// fails), the whole image is treated as a single lane — this keeps the
// call backward compatible with callers that don't know about lane
// detection yet, and keeps single-lane images working even if the strip
// model is ever missing/corrupt.
//
// process_tlc output: JSON string allocated with malloc (caller frees via
// free_result).
//   {
//     "spots": [ { "id", "rf", "intensity", "auc", "confidence", "box",
//                  "lane_id" }, ... ],   // flattened across all lanes,
//                                        // sorted by rf ascending
//     "plot_path": "...",
//     "count": N
//   }
// -----------------------------------------------------------------------

#if defined(_WIN32)
#define FFI_EXPORT __declspec(dllexport)
#else
#define FFI_EXPORT __attribute__((visibility("default")))
#endif

#include "SpotDetector.h"
#include "AUCCalculator.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "C++_Backend", __VA_ARGS__)
#else
#define LOGI(...) printf(__VA_ARGS__); printf("\n")
#endif

// Helper: split a string by a single-character delimiter

static std::vector<std::string> split_string(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper: parse manual_spots_str  "x1,y1,x2,y2;x1,y1,x2,y2;..."
// Coordinates are absolute image-pixel coordinates.
// Returns Spot objects with confidence=1.0, cls=0.

static std::vector<Spot> parse_manual_spots(const std::string& manual_str) {
    std::vector<Spot> spots;
    if (manual_str.empty()) return spots;

    auto entries = split_string(manual_str, ';');
    for (const auto& entry : entries) {
        if (entry.empty()) continue;
        auto coords = split_string(entry, ',');
        if (coords.size() < 4) continue;

        Spot s;
        s.x1 = std::stof(coords[0]);
        s.y1 = std::stof(coords[1]);
        s.x2 = std::stof(coords[2]);
        s.y2 = std::stof(coords[3]);
        s.confidence = 1.0f;
        s.cls = 0;
        spots.push_back(s);
    }
    return spots;
}

// Helper: compute the mean intensity of a grayscale ROI (absolute coords)

static double compute_mean_intensity(const cv::Mat& gray, float x1, float y1, float x2, float y2) {
    int ix1 = std::max(0, static_cast<int>(x1));
    int iy1 = std::max(0, static_cast<int>(y1));
    int ix2 = std::min(gray.cols, static_cast<int>(x2));
    int iy2 = std::min(gray.rows, static_cast<int>(y2));

    if (ix2 <= ix1 || iy2 <= iy1) return 0.0;

    cv::Mat roi = gray(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1));
    cv::Scalar mean_val = cv::mean(roi);
    return mean_val[0];
}


// Shared: Rf / intensity / AUC for one box (absolute coords), against the
// baseline/topline convention used throughout this file. Used by both
// process_tlc (for freshly-detected spots) and add_manual_spots (for a
// spot the user drew after the fact) so the two call paths can't drift.

struct SpotMetrics {
    double rf;
    double intensity;
    double auc;
};

static SpotMetrics compute_spot_metrics(
    const cv::Mat& gray,
    float x1, float y1, float x2, float y2,
    double baseline, double topline
) {
    double lane_rf_height = topline - baseline;
    if (std::abs(lane_rf_height) < 1.0) {
        lane_rf_height = (lane_rf_height < 0.0) ? -1.0 : 1.0;
    }

    SpotMetrics m;
    float center_y = (y1 + y2) / 2.0f;
    m.rf = (static_cast<double>(center_y) - baseline) / lane_rf_height;

    double mean_val = compute_mean_intensity(gray, x1, y1, x2, y2);
    m.intensity = 255.0 - mean_val;

    double area = (double)(x2 - x1) * (y2 - y1);
    std::vector<double> peak_x, peak_y;
    AUCCalculator::generate_peak_data(m.rf, m.intensity, area, peak_x, peak_y);
    m.auc = AUCCalculator::calculate_auc(peak_x, peak_y);

    return m;
}

// Helper: escape a string for safe JSON embedding

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}


// Lane geometry + polygon-overlap manual-spot matching
// (ported from the updated_backend CLI tool's multi-lane pipeline)

static std::vector<cv::Point> box_to_quad(double x1, double y1, double x2, double y2) {
    return {
        cv::Point((int)x1, (int)y1),
        cv::Point((int)x2, (int)y1),
        cv::Point((int)x2, (int)y2),
        cv::Point((int)x1, (int)y2)
    };
}

// Rasterizes both quads onto a local mask to compute their overlap
// percentage relative to the area of the smaller polygon.
static double polygon_overlap_percent(const std::vector<cv::Point>& poly_a, const std::vector<cv::Point>& poly_b) {
    std::vector<cv::Point> all_pts = poly_a;
    all_pts.insert(all_pts.end(), poly_b.begin(), poly_b.end());
    if (all_pts.empty()) return 0.0;

    cv::Rect bounding_rect = cv::boundingRect(all_pts);
    int min_x = bounding_rect.x;
    int min_y = bounding_rect.y;
    int w = std::max(bounding_rect.width + 2, 1);
    int h = std::max(bounding_rect.height + 2, 1);

    cv::Mat mask_a = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat mask_b = cv::Mat::zeros(h, w, CV_8UC1);

    std::vector<cv::Point> local_a(poly_a.size());
    for (size_t i = 0; i < poly_a.size(); ++i) {
        local_a[i] = cv::Point(poly_a[i].x - min_x, poly_a[i].y - min_y);
    }

    std::vector<cv::Point> local_b(poly_b.size());
    for (size_t i = 0; i < poly_b.size(); ++i) {
        local_b[i] = cv::Point(poly_b[i].x - min_x, poly_b[i].y - min_y);
    }

    std::vector<std::vector<cv::Point>> pts_a = { local_a };
    std::vector<std::vector<cv::Point>> pts_b = { local_b };

    cv::fillPoly(mask_a, pts_a, cv::Scalar(1));
    cv::fillPoly(mask_b, pts_b, cv::Scalar(1));

    double area_a = cv::countNonZero(mask_a);
    double area_b = cv::countNonZero(mask_b);
    double min_area = std::min(area_a, area_b);
    if (min_area == 0.0) return 0.0;

    cv::Mat intersection_mask;
    cv::bitwise_and(mask_a, mask_b, intersection_mask);
    double intersection = cv::countNonZero(intersection_mask);

    return (intersection / min_area) * 100.0;
}

// Lane-adaptive overlap threshold: mean + 0.5 * std of non-zero overlaps.
static double calculate_dynamic_threshold(const std::vector<double>& overlap_values, double min_thresh, double max_thresh) {
    std::vector<double> nonzero;
    for (double v : overlap_values) {
        if (v > 0.0) nonzero.push_back(v);
    }
    if (nonzero.empty()) return min_thresh;

    double sum = 0.0;
    for (double v : nonzero) sum += v;
    double mean = sum / nonzero.size();

    double sq_sum = 0.0;
    for (double v : nonzero) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / nonzero.size());

    double thresh = mean + 0.5 * std_dev;
    return std::max(min_thresh, std::min(thresh, max_thresh));
}

struct MergeResult {
    std::vector<Spot> spots;
    std::vector<bool> confirmed_flags; // true = matched a manual box, or manual-only; bypasses filtration
};

 /*Compares manual boxes (already in this lane's local coordinate space)
 against this lane's raw detections. A manual box that overlaps a
 detection above the dynamic threshold "confirms" that detection (forces
 it through filtration even if it would otherwise be rejected); a manual
 box with no good match is inserted as its own manual-only spot. Both
 outcomes bypass filtration entirely, since the user explicitly drew
 them — that's what fixes today's "manual spot silently doesn't appear"
 behaviour, where manual boxes used to be re-filtered exactly like any
 automatic detection.*/
static MergeResult merge_manual_and_detected_spots(
    const std::vector<Spot>& manual_spots_local,
    const std::vector<Spot>& detected_spots
) {
    std::vector<std::vector<cv::Point>> manual_quads;
    manual_quads.reserve(manual_spots_local.size());
    for (const auto& m : manual_spots_local) {
        manual_quads.push_back(box_to_quad(m.x1, m.y1, m.x2, m.y2));
    }

    std::vector<std::vector<cv::Point>> detected_quads;
    detected_quads.reserve(detected_spots.size());
    for (const auto& s : detected_spots) {
        detected_quads.push_back(box_to_quad(s.x1, s.y1, s.x2, s.y2));
    }

    std::vector<double> all_overlaps;
    all_overlaps.reserve(manual_quads.size() * detected_quads.size());
    std::vector<std::vector<double>> overlap_matrix(manual_quads.size(), std::vector<double>(detected_quads.size(), 0.0));

    for (size_t i = 0; i < manual_quads.size(); ++i) {
        for (size_t j = 0; j < detected_quads.size(); ++j) {
            double ov = polygon_overlap_percent(manual_quads[i], detected_quads[j]);
            overlap_matrix[i][j] = ov;
            all_overlaps.push_back(ov);
        }
    }

    double dynamic_threshold = calculate_dynamic_threshold(all_overlaps, 30.0, 85.0);

    std::vector<Spot> final_spots = detected_spots;
    std::vector<bool> confirmed_flags(final_spots.size(), false);

    for (size_t i = 0; i < manual_quads.size(); ++i) {
        int best_j = -1;
        double best_overlap = 0.0;

        if (!detected_quads.empty()) {
            best_overlap = overlap_matrix[i][0];
            best_j = 0;
            for (size_t j = 1; j < detected_quads.size(); ++j) {
                if (overlap_matrix[i][j] > best_overlap) {
                    best_overlap = overlap_matrix[i][j];
                    best_j = (int)j;
                }
            }
        }

        if (best_overlap >= dynamic_threshold && best_j != -1) {
            confirmed_flags[best_j] = true;
            LOGI("Manual spot #%d matched detection #%d (overlap=%.1f%%, threshold=%.1f%%)",
                 (int)(i + 1), best_j + 1, best_overlap, dynamic_threshold);
        } else {
            final_spots.push_back(manual_spots_local[i]);
            confirmed_flags.push_back(true);
            LOGI("Manual spot #%d added as manual-only (best overlap=%.1f%%, threshold=%.1f%%)",
                 (int)(i + 1), best_overlap, dynamic_threshold);
        }
    }

    MergeResult res;
    res.spots = final_spots;
    res.confirmed_flags = confirmed_flags;
    return res;
}

// Lane detection


struct Lane {
    int id;
    double x1, y1, x2, y2;
    cv::Mat crop;
};

 /*Detects lanes with the strip model, sorts them left-to-right, and always
 returns at least one lane — falling back to "whole image = one lane" if
 the model is unavailable, fails to load, or detects nothing. This is
 what keeps single-lane images (and callers that don't pass a strip
 model at all) working exactly as before.*/

static std::vector<Lane> detect_lanes(const cv::Mat& image, const std::string& strip_model_path) {
    std::vector<Lane> lanes;

    if (!strip_model_path.empty()) {
        try {
            SpotDetector strip_detector(strip_model_path);
            std::vector<Spot> lane_detections = strip_detector.detect(image, 0.25f, 0.45f);
            LOGI("Lane detection returned %d candidate(s).", static_cast<int>(lane_detections.size()));

            for (const auto& det : lane_detections) {
                double width = det.x2 - det.x1;
                if (width < 20.0) {
                    continue;
                }

                Lane lane;
                lane.id = 0;
                lane.x1 = det.x1;
                lane.y1 = det.y1;
                lane.x2 = det.x2;
                lane.y2 = det.y2;

                int ix1 = std::max(0, std::min((int)lane.x1, image.cols - 1));
                int iy1 = std::max(0, std::min((int)lane.y1, image.rows - 1));
                int ix2 = std::max(0, std::min((int)lane.x2, image.cols));
                int iy2 = std::max(0, std::min((int)lane.y2, image.rows));

                if (ix2 > ix1 && iy2 > iy1) {
                    lane.crop = image(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1)).clone();
                    lanes.push_back(lane);
                }
            }
        } catch (const std::exception& e) {
            LOGI("Lane detection failed (%s) — falling back to single-lane mode.", e.what());
            lanes.clear();
        }
    }

    std::sort(lanes.begin(), lanes.end(), [](const Lane& a, const Lane& b) {
        return a.x1 < b.x1;
    });
    for (size_t i = 0; i < lanes.size(); ++i) {
        lanes[i].id = (int)(i + 1);
    }

    if (lanes.empty()) {
        Lane lane;
        lane.id = 1;
        lane.x1 = 0;
        lane.y1 = 0;
        lane.x2 = image.cols;
        lane.y2 = image.rows;
        lane.crop = image.clone();
        lanes.push_back(lane);
    }

    return lanes;
}

 /*Assigns each manual (absolute-coordinate) spot to the lane whose
 horizontal span contains its center — or the nearest lane by horizontal
 distance if none does — and converts it into that lane's local
 coordinate space, clamped to the crop bounds.*/

static void assign_manual_spots_to_lanes(
    const std::vector<Spot>& manual_spots_absolute,
    const std::vector<Lane>& lanes,
    std::vector<std::vector<Spot>>& out_by_lane
) {
    for (const auto& m : manual_spots_absolute) {
        double center_x = (m.x1 + m.x2) / 2.0;

        int best_idx = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < lanes.size(); ++i) {
            const auto& lane = lanes[i];
            double dist;
            if (center_x >= lane.x1 && center_x <= lane.x2) {
                dist = 0.0;
            } else {
                dist = std::min(std::abs(center_x - lane.x1), std::abs(center_x - lane.x2));
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (int)i;
            }
        }

        const auto& lane = lanes[best_idx];
        Spot local;
        local.x1 = (float)std::max(0.0, std::min((double)(m.x1 - lane.x1), (double)lane.crop.cols));
        local.y1 = (float)std::max(0.0, std::min((double)(m.y1 - lane.y1), (double)lane.crop.rows));
        local.x2 = (float)std::max(0.0, std::min((double)(m.x2 - lane.x1), (double)lane.crop.cols));
        local.y2 = (float)std::max(0.0, std::min((double)(m.y2 - lane.y1), (double)lane.crop.rows));
        local.confidence = m.confidence;
        local.cls = m.cls;

        if (local.x2 > local.x1 && local.y2 > local.y1) {
            out_by_lane[best_idx].push_back(local);
        }
    }
}

// Struct to hold per-spot analysis results for sorting / output

struct SpotResult {
    int    id;
    int    lane_id;
    double rf;
    double intensity;
    double auc;
    float  confidence;
    float  x1, y1, x2, y2; // absolute image-pixel coordinates
};

// Exported C function: process_tlc

extern "C" FFI_EXPORT
const char* process_tlc(const char* json_args_str) {
    try {

        // Parse pipe-delimited input
        std::string input(json_args_str);
        auto parts = split_string(input, '|');

        // We expect up to 7 fields; pad with empty strings if fewer
        // (strip_model_path is optional — see file header).
        while (parts.size() < 7) parts.push_back("");

        std::string image_path       = parts[0];
        std::string model_path       = parts[1];
        double      baseline         = std::stod(parts[2]);
        double      topline          = std::stod(parts[3]);
        std::string plot_output_path = parts[4];
        std::string manual_spots_str = parts[5];
        std::string strip_model_path = parts[6];

        //Load the image
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            const char* err = "{\"error\":\"Failed to load image\"}";
            char* result = static_cast<char*>(std::malloc(std::strlen(err) + 1));
            std::strcpy(result, err);
            return result;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Detect lanes (always returns >= 1 lane; see detect_lanes)
        std::vector<Lane> lanes = detect_lanes(image, strip_model_path);
        LOGI("Active lanes: %d", static_cast<int>(lanes.size()));

        //Parse manual spots (absolute coords) and assign to lanes
        std::vector<Spot> manual_spots_absolute = parse_manual_spots(manual_spots_str);
        std::vector<std::vector<Spot>> manual_spots_by_lane(lanes.size());
        assign_manual_spots_to_lanes(manual_spots_absolute, lanes, manual_spots_by_lane);

         /*Spot detection + merge + filtration, per lane
         One SpotDetector instance is reused across all lanes within
         this single call — safe because the session never outlives
         process_tlc() (see SpotDetector.h for why that boundary matters
         on Android ARM64).*/
        SpotDetector spot_detector(model_path);

        std::vector<SpotResult> results;

        for (const auto& lane : lanes) {
            std::vector<Spot> auto_spots = spot_detector.detect(lane.crop, 0.0009f, 0.45f);

            MergeResult merged = merge_manual_and_detected_spots(
                manual_spots_by_lane[lane.id - 1], auto_spots);

             /*Filtration (resolution-aware, scoped to this lane's crop):
               - Area filter:       0.01% <= area <= 25% of the lane's area
               - Position filter:   |center_x - lane_center_x| < 0.45 * lane_width
               - Confidence filter: confidence >= 0.0009
             Manual/confirmed spots bypass all three — the user already
             told us they're real.*/
            double lane_width  = lane.crop.cols;
            double lane_center = lane_width / 2.0;
            double lane_area   = static_cast<double>(lane.crop.cols) * lane.crop.rows;
            double min_area    = lane_area * 0.0001;
            double max_area    = lane_area * 0.25;

            std::vector<Spot> filtered;
            for (size_t i = 0; i < merged.spots.size(); ++i) {
                const auto& s = merged.spots[i];

                if (merged.confirmed_flags[i]) {
                    filtered.push_back(s);
                    continue;
                }

                float area = (s.x2 - s.x1) * (s.y2 - s.y1);
                if (area < min_area || area > max_area) continue;
                if (s.confidence < 0.0009f) continue;

                float center_x = (s.x1 + s.x2) / 2.0f;
                if (std::abs(center_x - lane_center) >= 0.45 * lane_width) continue;

                filtered.push_back(s);
            }

            LOGI("Lane %d: %d auto + %d manual -> %d survived filtration.",
                 lane.id, static_cast<int>(auto_spots.size()),
                 static_cast<int>(manual_spots_by_lane[lane.id - 1].size()),
                 static_cast<int>(filtered.size()));

             /*Rf / intensity / AUC — computed against the *absolute* image
             and the single baseline/topline the caller supplied, exactly
             like the pre-multi-lane pipeline (lane cropping only changes
             where spots are *found*, not how Rf is defined).*/
            for (const auto& s : filtered) {
                float abs_x1 = s.x1 + (float)lane.x1;
                float abs_y1 = s.y1 + (float)lane.y1;
                float abs_x2 = s.x2 + (float)lane.x1;
                float abs_y2 = s.y2 + (float)lane.y1;

                SpotMetrics metrics = compute_spot_metrics(
                    gray, abs_x1, abs_y1, abs_x2, abs_y2, baseline, topline);

                SpotResult r;
                r.lane_id = lane.id;
                r.rf = metrics.rf;
                r.intensity = metrics.intensity;
                r.auc = metrics.auc;
                r.confidence = s.confidence;
                r.x1 = abs_x1;
                r.y1 = abs_y1;
                r.x2 = abs_x2;
                r.y2 = abs_y2;

                results.push_back(r);
            }
        }

        //Sort by Rf ascending across all lanes, assign IDs
        std::sort(results.begin(), results.end(),
                  [](const SpotResult& a, const SpotResult& b) {
                      return a.rf < b.rf;
                  });
        for (int i = 0; i < static_cast<int>(results.size()); ++i) {
            results[i].id = i + 1;
        }

        //Draw lane outlines + spot boxes on the original image
        static const cv::Scalar lane_colors[] = {
            cv::Scalar(255, 128, 0), cv::Scalar(0, 200, 255), cv::Scalar(200, 0, 255),
            cv::Scalar(255, 0, 128), cv::Scalar(128, 255, 0),
        };
        if (lanes.size() > 1) {
            for (const auto& lane : lanes) {
                cv::rectangle(image,
                              cv::Point((int)lane.x1, (int)lane.y1),
                              cv::Point((int)lane.x2, (int)lane.y2),
                              lane_colors[(lane.id - 1) % 5], 1);
            }
        }

        for (const auto& r : results) {
            cv::rectangle(image, cv::Point(static_cast<int>(r.x1), static_cast<int>(r.y1)),
                          cv::Point(static_cast<int>(r.x2), static_cast<int>(r.y2)),
                          cv::Scalar(0, 255, 0), 2);

            char label[64];
            std::snprintf(label, sizeof(label), "%d Rf:%.2f", r.id, r.rf);

            int text_baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &text_baseline);
            cv::Point textOrg(static_cast<int>(r.x1), static_cast<int>(r.y1) - 5);

            cv::rectangle(image,
                          textOrg + cv::Point(0, text_baseline),
                          textOrg + cv::Point(textSize.width, -textSize.height),
                          cv::Scalar(0, 255, 0),
                          cv::FILLED);

            cv::putText(image, label,
                        textOrg,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
        cv::imwrite(image_path, image);

        //Build JSON result string manually
        std::ostringstream json;
        json << std::fixed << std::setprecision(4);

        json << "{\"spots\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            if (i > 0) json << ",";
            json << "{"
                 << "\"id\":" << r.id << ","
                 << "\"lane_id\":" << r.lane_id << ","
                 << "\"rf\":" << r.rf << ","
                 << "\"intensity\":" << r.intensity << ","
                 << "\"auc\":" << r.auc << ","
                 << "\"confidence\":" << r.confidence << ","
                 << "\"box\":["
                 << r.x1 << "," << r.y1 << ","
                 << r.x2 << "," << r.y2
                 << "]"
                 << "}";
        }
        json << "],";
        json << "\"plot_path\":\"" << json_escape(plot_output_path) << "\",";
        json << "\"count\":" << results.size();
        json << "}";

        //Allocate result with malloc and return
        std::string json_str = json.str();
        char* result = static_cast<char*>(std::malloc(json_str.size() + 1));
        if (result) {
            std::memcpy(result, json_str.c_str(), json_str.size() + 1);
        }
        return result;

    } catch (const std::exception& e) {
        // Return error JSON on any exception
        std::string err = "{\"error\":\"" + json_escape(e.what()) + "\"}";
        char* result = static_cast<char*>(std::malloc(err.size() + 1));
        if (result) {
            std::memcpy(result, err.c_str(), err.size() + 1);
        }
        return result;
    }
}


/* Helper: parse existing_spots_str for add_manual_spots
   "x1,y1,x2,y2,rf,intensity,auc,confidence;..."
 These are trusted as given — no recompute, no filtration — since they
 were already confirmed by an earlier process_tlc() call.*/
static std::vector<SpotResult> parse_existing_spots(const std::string& s) {
    std::vector<SpotResult> spots;
    if (s.empty()) return spots;

    auto entries = split_string(s, ';');
    for (const auto& entry : entries) {
        if (entry.empty()) continue;
        auto f = split_string(entry, ',');
        if (f.size() < 8) continue;

        SpotResult r;
        r.id = 0;
        r.lane_id = (f.size() >= 9) ? std::stoi(f[8]) : 1;
        r.x1 = std::stof(f[0]);
        r.y1 = std::stof(f[1]);
        r.x2 = std::stof(f[2]);
        r.y2 = std::stof(f[3]);
        r.rf = std::stod(f[4]);
        r.intensity = std::stod(f[5]);
        r.auc = std::stod(f[6]);
        r.confidence = std::stof(f[7]);
        spots.push_back(r);
    }
    return spots;
}

/* Exported C function: add_manual_spots

 Adds one or more user-drawn spots to an already-processed image without
 re-running any ONNX model. Input format (pipe-delimited string):
   original_image_path|output_image_path|baseline|topline|existing_spots_str|new_boxes_str

 existing_spots_str: "x1,y1,x2,y2,rf,intensity,auc,confidence;..." — the
   caller's current spot list (in absolute original-image-pixel
   coordinates), redrawn as-is with no recomputation and no filtration.
 new_boxes_str: "x1,y1,x2,y2;..." — newly drawn boxes (same format as
   process_tlc's manual_spots_str). Rf/intensity/AUC are computed fresh
   for these; like any other manually-confirmed spot elsewhere in this
   backend, they are never filtered — the user explicitly marked them.

 Both existing and new spots are combined, sorted by Rf ascending, and
 renumbered from scratch, then drawn together onto a fresh copy of the
 *original* (clean) image — never onto the already-annotated one. That
 matters: if a new spot's Rf falls between two existing ones, the old
 spots' baked-in ID labels would otherwise go stale.

 Output JSON has the same shape as process_tlc's "spots" array, so the
 existing NewTlcResult/NewTlcSpot Dart parsing code is reused unchanged.*/

extern "C" FFI_EXPORT
const char* add_manual_spots(const char* json_args_str) {
    try {
        std::string input(json_args_str);
        auto parts = split_string(input, '|');
        while (parts.size() < 6) parts.push_back("");

        std::string original_image_path = parts[0];
        std::string output_image_path   = parts[1];
        double      baseline            = std::stod(parts[2]);
        double      topline             = std::stod(parts[3]);
        std::string existing_spots_str  = parts[4];
        std::string new_boxes_str       = parts[5];

        cv::Mat image = cv::imread(original_image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            const char* err = "{\"error\":\"Failed to load image\"}";
            char* result = static_cast<char*>(std::malloc(std::strlen(err) + 1));
            std::strcpy(result, err);
            return result;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<SpotResult> results = parse_existing_spots(existing_spots_str);

        // Build lane X-boundaries from existing spots so we can assign
        // new spots to the correct lane without re-running lane detection.
        // Map: lane_id -> (min_x1, max_x2) across all existing spots in that lane.
        std::map<int, std::pair<float, float>> lane_bounds;
        for (const auto& es : results) {
            auto it = lane_bounds.find(es.lane_id);
            if (it == lane_bounds.end()) {
                lane_bounds[es.lane_id] = {es.x1, es.x2};
            } else {
                it->second.first  = std::min(it->second.first,  es.x1);
                it->second.second = std::max(it->second.second, es.x2);
            }
        }

        // New spots: compute metrics fresh, never filtered.
        std::vector<Spot> new_boxes = parse_manual_spots(new_boxes_str);
        for (const auto& nb : new_boxes) {
            SpotMetrics metrics = compute_spot_metrics(
                gray, nb.x1, nb.y1, nb.x2, nb.y2, baseline, topline);

            SpotResult r;
            // Assign to the lane whose X-span contains the new spot's center,
            // or the nearest lane by horizontal distance.
            float center_x = (nb.x1 + nb.x2) / 2.0f;
            int best_lane = 1;
            float best_dist = std::numeric_limits<float>::max();
            for (const auto& lb : lane_bounds) {
                float dist;
                if (center_x >= lb.second.first && center_x <= lb.second.second) {
                    dist = 0.0f;
                } else {
                    dist = std::min(std::abs(center_x - lb.second.first),
                                    std::abs(center_x - lb.second.second));
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_lane = lb.first;
                }
            }
            r.lane_id = best_lane;
            r.rf = metrics.rf;
            r.intensity = metrics.intensity;
            r.auc = metrics.auc;
            r.confidence = 1.0f;
            r.x1 = nb.x1;
            r.y1 = nb.y1;
            r.x2 = nb.x2;
            r.y2 = nb.y2;

            results.push_back(r);
        }

        LOGI("add_manual_spots: %d existing + %d new = %d total.",
             static_cast<int>(results.size() - new_boxes.size()),
             static_cast<int>(new_boxes.size()),
             static_cast<int>(results.size()));

        std::sort(results.begin(), results.end(),
                  [](const SpotResult& a, const SpotResult& b) {
                      return a.rf < b.rf;
                  });
        for (int i = 0; i < static_cast<int>(results.size()); ++i) {
            results[i].id = i + 1;
        }

        // Draw every box + label fresh onto the clean original image.
        for (const auto& r : results) {
            cv::rectangle(image, cv::Point(static_cast<int>(r.x1), static_cast<int>(r.y1)),
                          cv::Point(static_cast<int>(r.x2), static_cast<int>(r.y2)),
                          cv::Scalar(0, 255, 0), 2);

            char label[64];
            std::snprintf(label, sizeof(label), "%d Rf:%.2f", r.id, r.rf);

            int text_baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &text_baseline);
            cv::Point textOrg(static_cast<int>(r.x1), static_cast<int>(r.y1) - 5);

            cv::rectangle(image,
                          textOrg + cv::Point(0, text_baseline),
                          textOrg + cv::Point(textSize.width, -textSize.height),
                          cv::Scalar(0, 255, 0),
                          cv::FILLED);

            cv::putText(image, label,
                        textOrg,
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
        cv::imwrite(output_image_path, image);

        // Build JSON result — same "spots" shape as process_tlc.
        std::ostringstream json;
        json << std::fixed << std::setprecision(4);

        json << "{\"spots\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            if (i > 0) json << ",";
            json << "{"
                 << "\"id\":" << r.id << ","
                 << "\"lane_id\":" << r.lane_id << ","
                 << "\"rf\":" << r.rf << ","
                 << "\"intensity\":" << r.intensity << ","
                 << "\"auc\":" << r.auc << ","
                 << "\"confidence\":" << r.confidence << ","
                 << "\"box\":["
                 << r.x1 << "," << r.y1 << ","
                 << r.x2 << "," << r.y2
                 << "]"
                 << "}";
        }
        json << "],";
        json << "\"count\":" << results.size();
        json << "}";

        std::string json_str = json.str();
        char* result = static_cast<char*>(std::malloc(json_str.size() + 1));
        if (result) {
            std::memcpy(result, json_str.c_str(), json_str.size() + 1);
        }
        return result;

    } catch (const std::exception& e) {
        std::string err = "{\"error\":\"" + json_escape(e.what()) + "\"}";
        char* result = static_cast<char*>(std::malloc(err.size() + 1));
        if (result) {
            std::memcpy(result, err.c_str(), err.size() + 1);
        }
        return result;
    }
}

// Exported C function: free_result
// Frees a string previously returned by process_tlc or add_manual_spots.
extern "C" FFI_EXPORT
void free_result(const char* ptr) {
    if (ptr) {
        std::free(const_cast<char*>(ptr));
    }
}
