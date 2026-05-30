#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class PreprocessModule {
public:
    cv::Mat gaussian_blur(const cv::Mat& src, int kernel_size = 5, double sigma = 0);
    cv::Mat canny_edges(const cv::Mat& src, double low = 50, double high = 150);
    cv::Mat harris_corners(const cv::Mat& src, cv::Mat& response,
                           int block_size = 2, int ksize = 3,
                           double k = 0.04, double thresh = 0.01);
    cv::Mat hog_visualise(const cv::Mat& src);
    cv::Mat letterbox(const cv::Mat& src, int target_size,
                      float& scale_x, float& scale_y, int& pad_x, int& pad_y);
};

} // namespace pipeline
