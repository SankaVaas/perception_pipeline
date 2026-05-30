#include "kalman_tracker.hpp"
#include <algorithm>

namespace pipeline {

static cv::Mat to_meas(const cv::Rect& r) {
    cv::Mat m = cv::Mat::zeros(4,1,CV_32F);
    m.at<float>(0) = r.x + r.width/2.f;
    m.at<float>(1) = r.y + r.height/2.f;
    m.at<float>(2) = (float)r.width;
    m.at<float>(3) = (float)r.height;
    return m;
}

KalmanTrack::KalmanTrack(const cv::Rect& bbox, int id)
    : kf_(8,4,0,CV_32F), id_(id) {
    auto& F = kf_.transitionMatrix;
    F = cv::Mat::eye(8,8,CV_32F);
    F.at<float>(0,4)=1; F.at<float>(1,5)=1; F.at<float>(2,6)=1; F.at<float>(3,7)=1;
    kf_.measurementMatrix = cv::Mat::zeros(4,8,CV_32F);
    kf_.measurementMatrix.at<float>(0,0)=1; kf_.measurementMatrix.at<float>(1,1)=1;
    kf_.measurementMatrix.at<float>(2,2)=1; kf_.measurementMatrix.at<float>(3,3)=1;
    kf_.processNoiseCov = cv::Mat::eye(8,8,CV_32F)*1e-2f;
    kf_.measurementNoiseCov = cv::Mat::eye(4,4,CV_32F)*1e-1f;
    kf_.errorCovPost = cv::Mat::eye(8,8,CV_32F);
    for(int i=4;i<8;i++) kf_.errorCovPost.at<float>(i,i)=1e4f;
    kf_.statePost = cv::Mat::zeros(8,1,CV_32F);
    cv::Mat init = to_meas(bbox);
    for(int i=0;i<4;i++) kf_.statePost.at<float>(i) = init.at<float>(i);
}

void KalmanTrack::predict() {
    kf_.predict(); age_++; time_since_update_++;
    const auto& s = kf_.statePre;
    float cx=s.at<float>(0), cy=s.at<float>(1), w=s.at<float>(2), h=s.at<float>(3);
    predicted_bbox_ = cv::Rect((int)(cx-w/2),(int)(cy-h/2),(int)w,(int)h);
}

void KalmanTrack::update(const cv::Rect& bbox) {
    kf_.correct(to_meas(bbox)); hits_++; time_since_update_=0;
}

cv::Rect KalmanTrack::get_bbox() const {
    const auto& s = kf_.statePost;
    float cx=s.at<float>(0), cy=s.at<float>(1), w=s.at<float>(2), h=s.at<float>(3);
    return cv::Rect((int)(cx-w/2),(int)(cy-h/2),(int)w,(int)h);
}

KalmanTracker::KalmanTracker(int max_age, int min_hits, float iou_threshold)
    : max_age_(max_age), min_hits_(min_hits), iou_thresh_(iou_threshold) {}

float KalmanTracker::iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.empty()) return 0.f;
    float i = (float)inter.area();
    float u = a.area() + b.area() - i;
    return u > 0 ? i/u : 0.f;
}

void KalmanTracker::match(const std::vector<Detection>& dets,
                          std::vector<int>& md, std::vector<int>& mt,
                          std::vector<int>& ud, std::vector<int>& ut) {
    std::vector<bool> du(dets.size(),false), tu(tracks_.size(),false);
    for(int t=0;t<(int)tracks_.size();t++) {
        int bd=-1; float bi=iou_thresh_;
        for(int d=0;d<(int)dets.size();d++) {
            if(du[d]) continue;
            float v=iou(tracks_[t].get_predicted(), dets[d].bbox);
            if(v>bi){bi=v;bd=d;}
        }
        if(bd>=0){md.push_back(bd);mt.push_back(t);du[bd]=true;tu[t]=true;}
    }
    for(int d=0;d<(int)dets.size();d++) if(!du[d]) ud.push_back(d);
    for(int t=0;t<(int)tracks_.size();t++) if(!tu[t]) ut.push_back(t);
}

std::vector<Track> KalmanTracker::update(const std::vector<Detection>& dets) {
    frame_count_++;
    for(auto& t:tracks_) t.predict();
    std::vector<int> md,mt,ud,ut;
    match(dets,md,mt,ud,ut);
    for(int i=0;i<(int)md.size();i++) tracks_[mt[i]].update(dets[md[i]].bbox);
    for(int d:ud) tracks_.emplace_back(dets[d].bbox, next_id_++);
    tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),
        [&](const KalmanTrack& t){return t.time_since_update()>max_age_;}),tracks_.end());
    std::vector<Track> result;
    for(int i=0;i<(int)tracks_.size();i++) {
        const auto& kt=tracks_[i];
        if(kt.hits()<min_hits_ && frame_count_>min_hits_) continue;
        Track tr; tr.track_id=kt.id(); tr.age=kt.age(); tr.hits=kt.hits();
        tr.last_det.bbox=kt.get_bbox();
        for(int j=0;j<(int)mt.size();j++)
            if(mt[j]==i){tr.last_det.class_id=dets[md[j]].class_id;
                         tr.last_det.confidence=dets[md[j]].confidence;}
        result.push_back(tr);
    }
    return result;
}

} // namespace pipeline
