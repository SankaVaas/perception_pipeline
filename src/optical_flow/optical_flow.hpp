#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class OpticalFlowModule {
public:
    struct Config {
        int    max_corners        = 200;
        double quality_level      = 0.01;
        double min_distance       = 10.0;
        int    block_size         = 3;
        cv::Size win_size         = {21, 21};
        int    max_level          = 3;
        int    max_iter           = 30;
        double epsilon            = 0.01;
        float  min_flow_mag       = 1.0f;
        bool   draw_vectors       = true;
        bool   draw_points        = true;
        bool   redetect_on_loss   = true;
        int    redetect_threshold = 50;
    };

    OpticalFlowModule() = default;
    explicit OpticalFlowModule(const Config& cfg) : cfg_(cfg) {}

    cv::Mat process(const cv::Mat& frame);
    const std::vector<cv::Point2f>& tracked_points() const { return curr_pts_; }
    const std::vector<cv::Point2f>& flow_vectors()   const { return flow_; }
    cv::Point2f estimate_ego_motion() const;
    void reset();
    bool is_initialized() const { return initialized_; }

private:
    void detect_features(const cv::Mat& gray);
    void draw_flow(cv::Mat& frame) const;

    Config cfg_;
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_, curr_pts_, flow_;
    std::vector<uchar> status_;
    std::vector<float> err_;
    bool initialized_ = false;
    int  frame_count_ = 0;
};

} // namespace pipeline
