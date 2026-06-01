#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

struct PoseResult {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat   R;
    double    distance_m  = 0;
    double    yaw_deg     = 0;
    double    pitch_deg   = 0;
    double    roll_deg    = 0;
    bool      valid       = false;
};

struct ObjectSize {
    double width_m;
    double height_m;
};

class PoseEstimator {
public:
    explicit PoseEstimator(const CameraIntrinsics& intrinsics);

    // PnP from bbox + assumed object size
    PoseResult estimate(const cv::Rect& bbox,
                        const ObjectSize& object_size) const;

    // ── NEW: Depth-fused pose ─────────────────────────────────────────────
    // Uses stereo Z to back-compute the real object size, then runs PnP.
    // This removes the dependency on assumed object dimensions entirely.
    // stereo_z: depth in metres from the disparity map at the bbox centre
    // ──────────────────────────────────────────────────────────────────────
    FusedPose estimate_fused(const cv::Rect& bbox,
                             float stereo_z) const;

    void draw_axes(cv::Mat& frame, const PoseResult& pose,
                   const cv::Rect& bbox, float axis_length = 0.1f) const;

    void draw_fused_axes(cv::Mat& frame, const FusedPose& pose,
                         float axis_length = 0.1f) const;

    void draw_info(cv::Mat& frame, const PoseResult& pose,
                   const cv::Rect& bbox) const;

    void draw_fused_info(cv::Mat& frame, const FusedPose& pose,
                         const cv::Rect& bbox) const;

    void set_intrinsics(const CameraIntrinsics& intrinsics);
    static ObjectSize default_size(int class_id);

private:
    PoseResult run_pnp(const cv::Rect& bbox,
                       double obj_w, double obj_h) const;
    void extract_euler(const cv::Mat& R,
                       double& yaw, double& pitch, double& roll) const;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double  fx_ = 1, fy_ = 1, cx_ = 0, cy_ = 0; // cached for back-projection
};

} // namespace pipeline
