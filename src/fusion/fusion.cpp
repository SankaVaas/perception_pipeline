#include "fusion.hpp"

namespace pipeline {

static cv::Point3f sample_3d(const cv::Mat& cloud, const cv::Rect& bbox) {
    cv::Rect roi = bbox & cv::Rect(0,0,cloud.cols,cloud.rows);
    if(roi.empty()) return {0,0,-1};
    int pw=std::max(1,roi.width/4), ph=std::max(1,roi.height/4);
    cv::Rect patch(roi.x+roi.width/2-pw/2, roi.y+roi.height/2-ph/2, pw, ph);
    patch &= cv::Rect(0,0,cloud.cols,cloud.rows);
    std::vector<float> zvals;
    cv::Mat sub=cloud(patch);
    for(int r=0;r<sub.rows;r++) for(int c=0;c<sub.cols;c++) {
        float z=sub.at<cv::Vec3f>(r,c)[2];
        if(z>0.01f&&z<200.f) zvals.push_back(z);
    }
    if(zvals.empty()) return {0,0,-1};
    std::nth_element(zvals.begin(),zvals.begin()+zvals.size()/2,zvals.end());
    float z=zvals[zvals.size()/2];
    cv::Vec3f ctr=cloud.at<cv::Vec3f>(patch.y+patch.height/2, patch.x+patch.width/2);
    return {ctr[0],ctr[1],z};
}

void FusionModule::fuse(std::vector<Detection>& dets, const cv::Mat& cloud) {
    if(cloud.empty()) return;
    for(auto& d:dets) d.position_3d = sample_3d(cloud, d.bbox);
}

void FusionModule::fuse_tracks(std::vector<Track>& tracks, const cv::Mat& cloud) {
    if(cloud.empty()) return;
    for(auto& t:tracks) t.last_det.position_3d = sample_3d(cloud, t.last_det.bbox);
}

} // namespace pipeline
