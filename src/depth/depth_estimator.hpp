#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class DepthEstimator {
public:
    void init(int min_disp = 0, int num_disp = 64, int block_size = 11);
    cv::Mat compute_disparity(const cv::Mat& left_rect, const cv::Mat& right_rect);
    cv::Mat disparity_to_3d(const cv::Mat& disparity, const cv::Mat& Q);
    float sample_depth_at_bbox(const cv::Mat& depth_map, const cv::Rect& bbox);
    cv::Mat colourize_disparity(const cv::Mat& disparity);
private:
    cv::Ptr<cv::StereoSGBM> sgbm_;
};

} // namespace pipeline
