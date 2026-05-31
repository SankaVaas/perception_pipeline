#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

// ─────────────────────────────────────────────────────────────────────────────
// 6DOF Pose Estimation using solvePnP
//
// Given:
//   - 4 corners of a detected bounding box (2D image points)
//   - Known real-world size of the object (3D model points)
//   - Camera intrinsic matrix K
//
// Solves the Perspective-n-Point (PnP) problem:
//   find R (rotation) and t (translation) such that:
//   x_image = K * [R | t] * X_world
//
// Output: rvec (rotation vector, Rodrigues) + tvec (translation in metres)
//   - tvec[2] = depth (Z distance to object centre)
//   - rvec converted to rotation matrix gives object orientation
//
// Robotics uses:
//   - Robot arm: "the cup is 0.3m ahead, tilted 15 degrees" → grasp planning
//   - Navigation: "the door is 2m away at -5 degrees yaw"
//   - AR: overlay 3D model exactly on detected object
// ─────────────────────────────────────────────────────────────────────────────

struct PoseResult {
    cv::Vec3d rvec;         // rotation vector (Rodrigues)
    cv::Vec3d tvec;         // translation vector in metres
    cv::Mat   R;            // 3x3 rotation matrix
    double    distance_m;   // Euclidean distance to object centre
    double    yaw_deg;      // yaw angle in degrees
    double    pitch_deg;    // pitch angle in degrees
    double    roll_deg;     // roll angle in degrees
    bool      valid = false;
};

// Known object sizes (width x height in metres)
struct ObjectSize {
    double width_m;
    double height_m;
};

class PoseEstimator {
public:
    explicit PoseEstimator(const CameraIntrinsics& intrinsics);

    // Estimate pose from a bounding box.
    // object_size: real-world width/height of this object class in metres.
    // Returns PoseResult with valid=false if estimation fails.
    PoseResult estimate(const cv::Rect& bbox,
                        const ObjectSize& object_size) const;

    // Draw pose axes on the frame (X=red, Y=green, Z=blue)
    void draw_axes(cv::Mat& frame,
                   const PoseResult& pose,
                   const cv::Rect& bbox,
                   float axis_length = 0.1f) const;

    // Draw pose info as text overlay
    void draw_info(cv::Mat& frame,
                   const PoseResult& pose,
                   const cv::Rect& bbox) const;

    // Update intrinsics (e.g. after calibration loads)
    void set_intrinsics(const CameraIntrinsics& intrinsics);

    // Default object sizes by COCO class id (approximate, in metres)
    static ObjectSize default_size(int class_id);

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
};

} // namespace pipeline
