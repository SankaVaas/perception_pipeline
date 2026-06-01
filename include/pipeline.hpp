#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>

namespace pipeline {

struct Detection {
    int         class_id    = -1;
    float       confidence  = 0.f;
    cv::Rect    bbox;
    cv::Point3f position_3d;
};

struct Track {
    int         track_id    = -1;
    int         age         = 0;
    int         hits        = 0;
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
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Size image_size;
};

struct StereoCalibration {
    CameraIntrinsics left;
    CameraIntrinsics right;
    cv::Mat R, T, R1, R2, P1, P2, Q;
};

struct PipelineConfig {
    int    camera_id       = 0;
    int    frame_width     = 640;
    int    frame_height    = 480;
    double target_fps      = 30.0;
    std::string model_path = "models/yolov8n.onnx";
    float  conf_threshold  = 0.45f;
    float  nms_threshold   = 0.50f;
    int    input_size      = 640;
    int    max_age         = 10;
    int    min_hits        = 3;
    bool   show_depth_map  = true;
    bool   show_edges      = false;
    bool   show_hog        = false;
};

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

} // namespace pipeline

// Depth-assisted pose — combines solvePnP with stereo depth
struct FusedPose {
    cv::Vec3d    rvec;           // rotation (Rodrigues)
    cv::Vec3d    tvec;           // translation — tvec[2] replaced by stereo Z
    cv::Mat      R;              // 3x3 rotation matrix
    double       depth_stereo_m; // Z from stereo (authoritative)
    double       depth_pnp_m;    // Z from PnP (for comparison)
    double       depth_error_m;  // |stereo - pnp| — calibration quality metric
    double       yaw_deg;
    double       pitch_deg;
    double       roll_deg;
    cv::Point3d  position_cam;   // (X,Y,Z) in camera frame, Z from stereo
    bool         valid = false;
};
