#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace pipeline {

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

struct Detection {
    int         class_id   = -1;
    float       confidence = 0.f;
    cv::Rect    bbox;
    cv::Point3f position_3d;
};

struct Track {
    int         track_id = -1;
    int         age      = 0;
    int         hits     = 0;
    Detection   last_det;
    cv::Point3f velocity_3d;
};

struct FrameTimings {
    double capture_ms    = 0;
    double preprocess_ms = 0;
    double depth_ms      = 0;
    double detect_ms     = 0;
    double track_ms      = 0;
    double render_ms     = 0;
    double total_ms      = 0;
};

struct CameraIntrinsics {
    cv::Mat  camera_matrix;
    cv::Mat  dist_coeffs;
    cv::Size image_size;
};

struct StereoCalibration {
    CameraIntrinsics left;
    CameraIntrinsics right;
    cv::Mat R, T, R1, R2, P1, P2, Q;
};

struct PipelineConfig {
    int         camera_id      = 0;
    int         frame_width    = 640;
    int         frame_height   = 480;
    double      target_fps     = 30.0;
    std::string model_path     = "models/yolov8n.onnx";
    float       conf_threshold = 0.45f;
    float       nms_threshold  = 0.50f;
    int         input_size     = 640;
    int         max_age        = 10;
    int         min_hits       = 3;
    bool        show_depth_map = true;
    bool        show_edges     = false;
    bool        show_hog       = false;
};

// Depth-assisted fused pose
struct FusedPose {
    cv::Vec3d   rvec;
    cv::Vec3d   tvec;
    cv::Mat     R;
    double      depth_stereo_m  = 0;
    double      depth_pnp_m     = 0;
    double      depth_error_m   = 0;
    double      yaw_deg         = 0;
    double      pitch_deg       = 0;
    double      roll_deg        = 0;
    cv::Point3d position_cam;
    bool        valid           = false;
};

// Data passed from inference thread → render thread
struct InferenceResult {
    cv::Mat                frame;
    cv::Mat                depth_vis;
    std::vector<Detection> detections;
    std::vector<Track>     tracks;
    FrameTimings           timings;
};

// ─────────────────────────────────────────────────────────────────────────────
// Stopwatch
// ─────────────────────────────────────────────────────────────────────────────
class Stopwatch {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe latest-frame buffer
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class LatestFrameBuffer {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data_     = std::move(value);
            has_data_ = true;
        }
        cv_.notify_one();
    }

    bool pop(T& out, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock,
                std::chrono::milliseconds(timeout_ms),
                [this]{ return has_data_ || stopped_; }))
            return false;
        if (stopped_ && !has_data_) return false;
        out       = std::move(data_);
        has_data_ = false;
        return true;
    }

    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        cv_.notify_all();
    }

    bool is_stopped() const { return stopped_; }

private:
    T                       data_;
    bool                    has_data_ = false;
    bool                    stopped_  = false;
    std::mutex              mutex_;
    std::condition_variable cv_;
};

} // namespace pipeline
