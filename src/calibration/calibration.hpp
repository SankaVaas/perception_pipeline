#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class CalibrationModule {
public:
    bool load(const std::string& yaml_path, StereoCalibration& calib);
    bool save(const std::string& yaml_path, const StereoCalibration& calib);
    void build_rectify_maps(StereoCalibration& calib,
                            cv::Mat& map1L, cv::Mat& map2L,
                            cv::Mat& map1R, cv::Mat& map2R);
    cv::Mat rectify(const cv::Mat& frame, const cv::Mat& map1, const cv::Mat& map2);
    static StereoCalibration make_dummy(cv::Size image_size);
};

} // namespace pipeline
