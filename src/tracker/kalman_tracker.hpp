#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class KalmanTrack {
public:
    explicit KalmanTrack(const cv::Rect& bbox, int id);
    void predict();
    void update(const cv::Rect& bbox);
    cv::Rect get_bbox() const;
    cv::Rect get_predicted() const { return predicted_bbox_; }
    int id()  const { return id_; }
    int age() const { return age_; }
    int hits() const { return hits_; }
    int time_since_update() const { return time_since_update_; }
private:
    cv::KalmanFilter kf_;
    cv::Rect predicted_bbox_;
    int id_, age_=0, hits_=0, time_since_update_=0;
};

class KalmanTracker {
public:
    explicit KalmanTracker(int max_age=10, int min_hits=3, float iou_threshold=0.3f);
    std::vector<Track> update(const std::vector<Detection>& detections);
private:
    float iou(const cv::Rect& a, const cv::Rect& b);
    void match(const std::vector<Detection>& dets,
               std::vector<int>& md, std::vector<int>& mt,
               std::vector<int>& ud, std::vector<int>& ut);
    std::vector<KalmanTrack> tracks_;
    int next_id_=0, max_age_, min_hits_, frame_count_=0;
    float iou_thresh_;
};

} // namespace pipeline
