#include "pose_estimator.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace pipeline {

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────
PoseEstimator::PoseEstimator(const CameraIntrinsics& intrinsics) {
    set_intrinsics(intrinsics);
}

void PoseEstimator::set_intrinsics(const CameraIntrinsics& intrinsics) {
    camera_matrix_ = intrinsics.camera_matrix.clone();
    dist_coeffs_   = intrinsics.dist_coeffs.clone();
    if (camera_matrix_.type() != CV_64F)
        camera_matrix_.convertTo(camera_matrix_, CV_64F);
    if (dist_coeffs_.type() != CV_64F)
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);

    if (!camera_matrix_.empty() && camera_matrix_.rows == 3) {
        fx_ = camera_matrix_.at<double>(0,0);
        fy_ = camera_matrix_.at<double>(1,1);
        cx_ = camera_matrix_.at<double>(0,2);
        cy_ = camera_matrix_.at<double>(1,2);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Euler angles from rotation matrix (ZYX convention)
// ─────────────────────────────────────────────────────────────────────────────
void PoseEstimator::extract_euler(const cv::Mat& R,
                                  double& yaw, double& pitch, double& roll) const {
    double r00=R.at<double>(0,0), r10=R.at<double>(1,0), r20=R.at<double>(2,0);
    double r21=R.at<double>(2,1), r22=R.at<double>(2,2);
    yaw   = std::atan2(r10, r00)                            * 180.0/CV_PI;
    pitch = std::atan2(-r20, std::sqrt(r21*r21+r22*r22))    * 180.0/CV_PI;
    roll  = std::atan2(r21, r22)                             * 180.0/CV_PI;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core PnP runner (shared by both estimate() and estimate_fused())
// ─────────────────────────────────────────────────────────────────────────────
PoseResult PoseEstimator::run_pnp(const cv::Rect& bbox,
                                  double obj_w, double obj_h) const {
    PoseResult result;
    if (camera_matrix_.empty() || bbox.area() <= 0) return result;

    double hw = obj_w / 2.0, hh = obj_h / 2.0;
    std::vector<cv::Point3d> model_pts = {
        {-hw,-hh,0}, {hw,-hh,0}, {hw,hh,0}, {-hw,hh,0}
    };
    std::vector<cv::Point2d> image_pts = {
        {(double)bbox.x,              (double)bbox.y             },
        {(double)(bbox.x+bbox.width), (double)bbox.y             },
        {(double)(bbox.x+bbox.width), (double)(bbox.y+bbox.height)},
        {(double)bbox.x,              (double)(bbox.y+bbox.height)}
    };

    cv::Vec3d rvec, tvec;
    bool ok = cv::solvePnP(model_pts, image_pts,
                           camera_matrix_, dist_coeffs_,
                           rvec, tvec, false,
                           cv::SOLVEPNP_IPPE_SQUARE);
    if (!ok) return result;

    result.rvec  = rvec;
    result.tvec  = tvec;
    result.valid = true;
    result.distance_m = cv::norm(tvec);
    cv::Rodrigues(rvec, result.R);
    extract_euler(result.R, result.yaw_deg, result.pitch_deg, result.roll_deg);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard PnP (assumed object size)
// ─────────────────────────────────────────────────────────────────────────────
PoseResult PoseEstimator::estimate(const cv::Rect& bbox,
                                   const ObjectSize& obj) const {
    return run_pnp(bbox, obj.width_m, obj.height_m);
}

// ─────────────────────────────────────────────────────────────────────────────
// Depth-fused pose
//
// Concept: back-projection
// If we know Z (from stereo) and the pixel bbox size, we can compute the
// real-world object size using the pinhole camera model:
//
//   W_real = bbox_width_pixels  * Z / fx
//   H_real = bbox_height_pixels * Z / fy
//
// This is just the inverse of the projection equation:
//   x_pixel = fx * X_world / Z + cx
//   → X_world = (x_pixel - cx) * Z / fx
//
// Then we run PnP with these measured dimensions instead of assumed ones.
// The resulting tvec[2] (Z from PnP) should be close to stereo_z —
// the difference is a useful calibration quality metric.
// ─────────────────────────────────────────────────────────────────────────────
FusedPose PoseEstimator::estimate_fused(const cv::Rect& bbox,
                                        float stereo_z) const {
    FusedPose result;
    if (stereo_z <= 0.01f || stereo_z > 100.f) return result;
    if (camera_matrix_.empty() || bbox.area() <= 0) return result;

    // Back-project bbox size to real-world dimensions using stereo Z
    double Z    = static_cast<double>(stereo_z);
    double obj_w = bbox.width  * Z / fx_;
    double obj_h = bbox.height * Z / fy_;

    // Run PnP with the stereo-derived object size
    PoseResult pnp = run_pnp(bbox, obj_w, obj_h);
    if (!pnp.valid) return result;

    result.rvec        = pnp.rvec;
    result.R           = pnp.R;
    result.depth_pnp_m = pnp.tvec[2];          // Z from PnP
    result.depth_stereo_m = static_cast<double>(stereo_z);
    result.depth_error_m  = std::abs(result.depth_pnp_m - result.depth_stereo_m);

    // Replace PnP translation Z with the authoritative stereo depth
    // Back-project bbox centre to get X, Y in camera frame
    double u = bbox.x + bbox.width  / 2.0;
    double v = bbox.y + bbox.height / 2.0;
    result.position_cam.x = (u - cx_) * Z / fx_;
    result.position_cam.y = (v - cy_) * Z / fy_;
    result.position_cam.z = Z;

    // Build tvec from stereo-corrected position
    result.tvec = cv::Vec3d(result.position_cam.x,
                            result.position_cam.y,
                            result.position_cam.z);

    extract_euler(result.R, result.yaw_deg, result.pitch_deg, result.roll_deg);
    result.valid = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawing
// ─────────────────────────────────────────────────────────────────────────────
void PoseEstimator::draw_axes(cv::Mat& frame, const PoseResult& pose,
                              const cv::Rect& /*bbox*/, float axis_length) const {
    if (!pose.valid || camera_matrix_.empty()) return;

    std::vector<cv::Point3d> axis_pts = {
        {0,0,0}, {axis_length,0,0}, {0,axis_length,0}, {0,0,axis_length}
    };
    std::vector<cv::Point2d> projected;
    cv::projectPoints(axis_pts, pose.rvec, pose.tvec,
                      camera_matrix_, dist_coeffs_, projected);

    auto clamp = [&](cv::Point2d p) {
        return cv::Point(std::max(0,std::min((int)p.x,frame.cols-1)),
                         std::max(0,std::min((int)p.y,frame.rows-1)));
    };
    cv::Point o = clamp(projected[0]);
    cv::arrowedLine(frame,o,clamp(projected[1]),cv::Scalar(0,0,255),  2,cv::LINE_AA,0,0.2);
    cv::arrowedLine(frame,o,clamp(projected[2]),cv::Scalar(0,255,0),  2,cv::LINE_AA,0,0.2);
    cv::arrowedLine(frame,o,clamp(projected[3]),cv::Scalar(255,0,0),  2,cv::LINE_AA,0,0.2);
}

void PoseEstimator::draw_fused_axes(cv::Mat& frame, const FusedPose& pose,
                                    float axis_length) const {
    if (!pose.valid || camera_matrix_.empty()) return;

    std::vector<cv::Point3d> axis_pts = {
        {0,0,0}, {axis_length,0,0}, {0,axis_length,0}, {0,0,axis_length}
    };
    std::vector<cv::Point2d> projected;
    cv::projectPoints(axis_pts, pose.rvec, pose.tvec,
                      camera_matrix_, dist_coeffs_, projected);

    auto clamp = [&](cv::Point2d p) {
        return cv::Point(std::max(0,std::min((int)p.x,frame.cols-1)),
                         std::max(0,std::min((int)p.y,frame.rows-1)));
    };
    cv::Point o = clamp(projected[0]);
    // Thicker arrows to distinguish from standard pose
    cv::arrowedLine(frame,o,clamp(projected[1]),cv::Scalar(0,80,255), 3,cv::LINE_AA,0,0.2);
    cv::arrowedLine(frame,o,clamp(projected[2]),cv::Scalar(80,255,0), 3,cv::LINE_AA,0,0.2);
    cv::arrowedLine(frame,o,clamp(projected[3]),cv::Scalar(255,80,0), 3,cv::LINE_AA,0,0.2);
}

void PoseEstimator::draw_info(cv::Mat& frame, const PoseResult& pose,
                              const cv::Rect& bbox) const {
    if (!pose.valid) return;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "D:" << pose.distance_m << "m"
       << " Y:" << (int)pose.yaw_deg
       << " P:" << (int)pose.pitch_deg
       << " R:" << (int)pose.roll_deg;
    cv::Point tp(bbox.x, bbox.y-22);
    if (tp.y < 12) tp.y = bbox.y+bbox.height+16;
    cv::putText(frame, ss.str(), tp,
                cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(0,220,255), 1, cv::LINE_AA);
}

void PoseEstimator::draw_fused_info(cv::Mat& frame, const FusedPose& pose,
                                    const cv::Rect& bbox) const {
    if (!pose.valid) return;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "Z:" << pose.depth_stereo_m << "m"
       << " X:" << pose.position_cam.x
       << " Y:" << pose.position_cam.y;
    std::ostringstream ss2;
    ss2 << "Yaw:" << (int)pose.yaw_deg
        << " Pit:" << (int)pose.pitch_deg
        << " Rol:" << (int)pose.roll_deg
        << " err:" << std::fixed << std::setprecision(3) << pose.depth_error_m << "m";

    cv::Point tp(bbox.x, bbox.y-36);
    if (tp.y < 12) tp.y = bbox.y+bbox.height+16;
    cv::putText(frame, ss.str(),  tp,
                cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(0,255,200), 1, cv::LINE_AA);
    cv::putText(frame, ss2.str(), tp+cv::Point(0,14),
                cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(0,200,255), 1, cv::LINE_AA);
}

ObjectSize PoseEstimator::default_size(int class_id) {
    switch (class_id) {
        case 0:  return {0.50, 1.70};
        case 2:  return {4.50, 1.50};
        case 3:  return {2.00, 1.50};
        case 5:  return {12.0, 3.50};
        case 7:  return {8.00, 3.00};
        case 15: return {0.45, 0.35};
        case 16: return {0.60, 0.60};
        case 24: return {0.07, 0.20};
        case 26: return {0.10, 0.10};
        case 39: return {0.30, 0.30};
        case 41: return {0.06, 0.12};
        case 56: return {0.50, 0.90};
        case 63: return {0.38, 0.26};
        case 73: return {0.20, 0.28};
        default: return {0.30, 0.30};
    }
}

} // namespace pipeline
