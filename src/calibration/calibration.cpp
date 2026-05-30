#include "calibration.hpp"
#include <iostream>

namespace pipeline {

bool CalibrationModule::load(const std::string& yaml_path, StereoCalibration& calib) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Calibration] Could not open: " << yaml_path << " - using dummy.\n";
        return false;
    }
    fs["left_camera_matrix"]  >> calib.left.camera_matrix;
    fs["left_dist_coeffs"]    >> calib.left.dist_coeffs;
    fs["right_camera_matrix"] >> calib.right.camera_matrix;
    fs["right_dist_coeffs"]   >> calib.right.dist_coeffs;
    fs["R"] >> calib.R; fs["T"] >> calib.T;
    fs["R1"] >> calib.R1; fs["R2"] >> calib.R2;
    fs["P1"] >> calib.P1; fs["P2"] >> calib.P2;
    fs["Q"] >> calib.Q;
    int w = 640, h = 480;
    fs["image_width"] >> w; fs["image_height"] >> h;
    calib.left.image_size = calib.right.image_size = cv::Size(w, h);
    return true;
}

bool CalibrationModule::save(const std::string& yaml_path, const StereoCalibration& calib) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "left_camera_matrix" << calib.left.camera_matrix;
    fs << "left_dist_coeffs"   << calib.left.dist_coeffs;
    fs << "right_camera_matrix"<< calib.right.camera_matrix;
    fs << "right_dist_coeffs"  << calib.right.dist_coeffs;
    fs << "R" << calib.R << "T" << calib.T;
    fs << "R1" << calib.R1 << "R2" << calib.R2;
    fs << "P1" << calib.P1 << "P2" << calib.P2 << "Q" << calib.Q;
    fs << "image_width" << calib.left.image_size.width;
    fs << "image_height" << calib.left.image_size.height;
    return true;
}

void CalibrationModule::build_rectify_maps(StereoCalibration& calib,
                                           cv::Mat& map1L, cv::Mat& map2L,
                                           cv::Mat& map1R, cv::Mat& map2R) {
    const cv::Size sz = calib.left.image_size;
    if (calib.R1.empty()) {
        cv::stereoRectify(
            calib.left.camera_matrix,  calib.left.dist_coeffs,
            calib.right.camera_matrix, calib.right.dist_coeffs,
            sz, calib.R, calib.T,
            calib.R1, calib.R2, calib.P1, calib.P2, calib.Q,
            cv::CALIB_ZERO_DISPARITY, 0, sz);
    }
    cv::initUndistortRectifyMap(calib.left.camera_matrix, calib.left.dist_coeffs,
                                calib.R1, calib.P1, sz, CV_32FC1, map1L, map2L);
    cv::initUndistortRectifyMap(calib.right.camera_matrix, calib.right.dist_coeffs,
                                calib.R2, calib.P2, sz, CV_32FC1, map1R, map2R);
}

cv::Mat CalibrationModule::rectify(const cv::Mat& frame,
                                   const cv::Mat& map1, const cv::Mat& map2) {
    cv::Mat out;
    cv::remap(frame, out, map1, map2, cv::INTER_LINEAR);
    return out;
}

StereoCalibration CalibrationModule::make_dummy(cv::Size sz) {
    StereoCalibration calib;
    double fx = sz.width * 0.8;
    double fy = fx;
    double cx = sz.width  / 2.0;
    double cy = sz.height / 2.0;
    double data[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    calib.left.camera_matrix  = cv::Mat(3, 3, CV_64F, data).clone();
    calib.right.camera_matrix = calib.left.camera_matrix.clone();
    calib.left.dist_coeffs  = cv::Mat::zeros(1, 5, CV_64F);
    calib.right.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    calib.left.image_size = calib.right.image_size = sz;
    calib.R = cv::Mat::eye(3, 3, CV_64F);
    double t[3] = {-0.06, 0, 0};
    calib.T = cv::Mat(3, 1, CV_64F, t).clone();
    return calib;
}

} // namespace pipeline
