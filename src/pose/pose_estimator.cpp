#include "pose_estimator.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace pipeline {

PoseEstimator::PoseEstimator(const CameraIntrinsics& intrinsics) {
    set_intrinsics(intrinsics);
}

void PoseEstimator::set_intrinsics(const CameraIntrinsics& intrinsics) {
    camera_matrix_ = intrinsics.camera_matrix.clone();
    dist_coeffs_   = intrinsics.dist_coeffs.clone();

    // Ensure float64
    if (camera_matrix_.type() != CV_64F)
        camera_matrix_.convertTo(camera_matrix_, CV_64F);
    if (dist_coeffs_.type() != CV_64F)
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
}

PoseResult PoseEstimator::estimate(const cv::Rect& bbox,
                                   const ObjectSize& obj) const {
    PoseResult result;
    if (camera_matrix_.empty() || bbox.area() <= 0) return result;

    // ── Concept: 3D model points ──────────────────────────────────────────
    // We model the object as a flat rectangle centred at the origin.
    // The four corners in object (world) coordinates:
    //
    //   (-w/2, -h/2, 0)  ──────  (w/2, -h/2, 0)
    //        |                         |
    //   (-w/2,  h/2, 0)  ──────  (w/2,  h/2, 0)
    //
    // Z=0 means we assume the object face is perpendicular to the camera.
    // This is the simplest valid PnP model — 4 coplanar points.
    // ──────────────────────────────────────────────────────────────────────
    double hw = obj.width_m  / 2.0;
    double hh = obj.height_m / 2.0;

    std::vector<cv::Point3d> model_pts = {
        {-hw, -hh, 0},   // top-left
        { hw, -hh, 0},   // top-right
        { hw,  hh, 0},   // bottom-right
        {-hw,  hh, 0}    // bottom-left
    };

    // ── 2D image points from bounding box corners ─────────────────────────
    // We use the four bbox corners as correspondences to the model corners.
    // This assumes the bbox tightly fits the object face — a reasonable
    // approximation for frontal detections.
    // ──────────────────────────────────────────────────────────────────────
    std::vector<cv::Point2d> image_pts = {
        {(double)bbox.x,              (double)bbox.y             },  // top-left
        {(double)(bbox.x + bbox.width),(double)bbox.y            },  // top-right
        {(double)(bbox.x + bbox.width),(double)(bbox.y+bbox.height)},// bottom-right
        {(double)bbox.x,              (double)(bbox.y+bbox.height)}  // bottom-left
    };

    // ── solvePnP ──────────────────────────────────────────────────────────
    // Concept: solvePnP finds R and t minimising the reprojection error:
    //   sum_i || x_i - pi(K, R, t, X_i) ||^2
    //
    // IPPE_SQUARE is the best solver for coplanar square/rectangular targets
    // (Infinitesimal Plane-based Pose Estimation). It's faster and more
    // accurate than ITERATIVE for planar cases.
    // ──────────────────────────────────────────────────────────────────────
    cv::Vec3d rvec, tvec;
    bool ok = cv::solvePnP(model_pts, image_pts,
                           camera_matrix_, dist_coeffs_,
                           rvec, tvec,
                           false,
                           cv::SOLVEPNP_IPPE_SQUARE);

    if (!ok) return result;

    result.rvec  = rvec;
    result.tvec  = tvec;
    result.valid = true;

    // ── Distance ──────────────────────────────────────────────────────────
    result.distance_m = cv::norm(tvec);

    // ── Rotation matrix from Rodrigues vector ────────────────────────────
    // Concept: Rodrigues representation encodes a rotation as a vector
    // whose direction is the rotation axis and magnitude is the angle (rad).
    // cv::Rodrigues converts this to/from the 3x3 rotation matrix.
    // ──────────────────────────────────────────────────────────────────────
    cv::Rodrigues(rvec, result.R);

    // ── Euler angles (ZYX convention = yaw-pitch-roll) ────────────────────
    // Extract from rotation matrix:
    //   yaw   = atan2(R[1,0], R[0,0])
    //   pitch = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
    //   roll  = atan2(R[2,1], R[2,2])
    // ──────────────────────────────────────────────────────────────────────
    double r00 = result.R.at<double>(0,0);
    double r10 = result.R.at<double>(1,0);
    double r20 = result.R.at<double>(2,0);
    double r21 = result.R.at<double>(2,1);
    double r22 = result.R.at<double>(2,2);

    result.yaw_deg   = std::atan2(r10, r00)                           * 180.0 / CV_PI;
    result.pitch_deg = std::atan2(-r20, std::sqrt(r21*r21 + r22*r22)) * 180.0 / CV_PI;
    result.roll_deg  = std::atan2(r21, r22)                           * 180.0 / CV_PI;

    return result;
}

void PoseEstimator::draw_axes(cv::Mat& frame,
                              const PoseResult& pose,
                              const cv::Rect& bbox,
                              float axis_length) const {
    if (!pose.valid || camera_matrix_.empty()) return;

    // ── Project 3D axis endpoints onto the image ──────────────────────────
    // We draw three unit vectors from the object origin:
    //   X axis (red)   → (axis_length, 0, 0)
    //   Y axis (green) → (0, axis_length, 0)
    //   Z axis (blue)  → (0, 0, axis_length)  ← points toward camera
    // ──────────────────────────────────────────────────────────────────────
    std::vector<cv::Point3d> axis_pts = {
        {0, 0, 0},
        {axis_length, 0, 0},
        {0, axis_length, 0},
        {0, 0, axis_length}
    };

    std::vector<cv::Point2d> projected;
    cv::projectPoints(axis_pts, pose.rvec, pose.tvec,
                      camera_matrix_, dist_coeffs_, projected);

    // Clamp to frame
    auto clamp_pt = [&](cv::Point2d p) {
        return cv::Point(
            std::max(0, std::min((int)p.x, frame.cols-1)),
            std::max(0, std::min((int)p.y, frame.rows-1)));
    };

    cv::Point origin = clamp_pt(projected[0]);
    cv::arrowedLine(frame, origin, clamp_pt(projected[1]),
                    cv::Scalar(0,0,255),   2, cv::LINE_AA, 0, 0.2); // X red
    cv::arrowedLine(frame, origin, clamp_pt(projected[2]),
                    cv::Scalar(0,255,0),   2, cv::LINE_AA, 0, 0.2); // Y green
    cv::arrowedLine(frame, origin, clamp_pt(projected[3]),
                    cv::Scalar(255,0,0),   2, cv::LINE_AA, 0, 0.2); // Z blue
}

void PoseEstimator::draw_info(cv::Mat& frame,
                              const PoseResult& pose,
                              const cv::Rect& bbox) const {
    if (!pose.valid) return;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "D:" << pose.distance_m << "m"
       << " Y:" << (int)pose.yaw_deg
       << " P:" << (int)pose.pitch_deg
       << " R:" << (int)pose.roll_deg;

    cv::Point text_pos(bbox.x, bbox.y - 22);
    if (text_pos.y < 12) text_pos.y = bbox.y + bbox.height + 16;

    cv::putText(frame, ss.str(), text_pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
}

ObjectSize PoseEstimator::default_size(int class_id) {
    // Approximate real-world sizes for common COCO classes (metres)
    // Width x Height
    switch (class_id) {
        case 0:  return {0.50, 1.70};  // person (shoulder width x height)
        case 2:  return {4.50, 1.50};  // car
        case 3:  return {2.00, 1.50};  // motorcycle
        case 5:  return {12.0, 3.50};  // bus
        case 7:  return {8.00, 3.00};  // truck
        case 14: return {0.30, 0.30};  // bird
        case 15: return {0.45, 0.35};  // cat
        case 16: return {0.60, 0.60};  // dog
        case 24: return {0.07, 0.20};  // bottle
        case 26: return {0.10, 0.10};  // cup
        case 28: return {0.20, 0.03};  // knife
        case 39: return {0.30, 0.30};  // laptop (closed)
        case 41: return {0.06, 0.12};  // cell phone
        case 56: return {0.50, 0.90};  // chair
        case 57: return {1.80, 0.90};  // couch
        case 63: return {0.38, 0.26};  // laptop (open)
        case 67: return {0.06, 0.12};  // cell phone
        case 73: return {0.20, 0.28};  // book
        default: return {0.30, 0.30};  // generic 30cm object
    }
}

} // namespace pipeline
