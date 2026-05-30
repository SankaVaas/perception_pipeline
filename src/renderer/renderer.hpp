#pragma once
#include "../../include/pipeline.hpp"
#include "../detector/detector.hpp"

namespace pipeline {
class Renderer {
public:
    void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets, bool show_3d=true);
    void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks, bool show_3d=true);
    void draw_hud(cv::Mat& frame, const FrameTimings& t, int num_tracks);
    void draw_depth_overlay(cv::Mat& frame, const cv::Mat& depth_coloured, float alpha=0.4f);
private:
    cv::Scalar track_colour(int id);
};
} // namespace pipeline
