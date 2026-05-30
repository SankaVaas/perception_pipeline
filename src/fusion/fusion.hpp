#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {
class FusionModule {
public:
    void fuse(std::vector<Detection>& detections, const cv::Mat& depth_cloud);
    void fuse_tracks(std::vector<Track>& tracks, const cv::Mat& depth_cloud);
};
} // namespace pipeline
