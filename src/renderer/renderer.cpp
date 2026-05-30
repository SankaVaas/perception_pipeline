#include "renderer.hpp"
#include <sstream>
#include <iomanip>

namespace pipeline {

cv::Scalar Renderer::track_colour(int id) {
    float hue = std::fmod(id*137.508f, 360.f);
    cv::Mat hsv(1,1,CV_8UC3,cv::Scalar((int)(hue/2),200,220)), bgr;
    cv::cvtColor(hsv,bgr,cv::COLOR_HSV2BGR);
    auto px=bgr.at<cv::Vec3b>(0,0);
    return cv::Scalar(px[0],px[1],px[2]);
}

void Renderer::draw_detections(cv::Mat& frame, const std::vector<Detection>& dets, bool show_3d) {
    for(const auto& d:dets) {
        cv::Scalar col(0,255,80);
        cv::rectangle(frame,d.bbox,col,2);
        std::ostringstream label;
        const auto& names=Detector::coco_names();
        if(d.class_id>=0&&d.class_id<(int)names.size()) label<<names[d.class_id];
        label<<" "<<std::fixed<<std::setprecision(2)<<d.confidence;
        if(show_3d&&d.position_3d.z>0)
            label<<" | "<<std::setprecision(1)<<d.position_3d.z<<"m";
        cv::putText(frame,label.str(),cv::Point(d.bbox.x,d.bbox.y-6),
                    cv::FONT_HERSHEY_SIMPLEX,0.5,col,1,cv::LINE_AA);
    }
}

void Renderer::draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks, bool show_3d) {
    for(const auto& t:tracks) {
        cv::Scalar col=track_colour(t.track_id);
        cv::rectangle(frame,t.last_det.bbox,col,2);
        std::ostringstream label;
        label<<"#"<<t.track_id;
        const auto& names=Detector::coco_names();
        if(t.last_det.class_id>=0&&t.last_det.class_id<(int)names.size())
            label<<" "<<names[t.last_det.class_id];
        if(show_3d&&t.last_det.position_3d.z>0)
            label<<" "<<std::fixed<<std::setprecision(1)<<t.last_det.position_3d.z<<"m";
        cv::putText(frame,label.str(),cv::Point(t.last_det.bbox.x,t.last_det.bbox.y-6),
                    cv::FONT_HERSHEY_SIMPLEX,0.5,col,1,cv::LINE_AA);
    }
}

void Renderer::draw_hud(cv::Mat& frame, const FrameTimings& t, int num_tracks) {
    auto put=[&](const std::string& s,int row){
        cv::putText(frame,s,cv::Point(10,20+row*18),
                    cv::FONT_HERSHEY_SIMPLEX,0.45,cv::Scalar(220,220,220),1,cv::LINE_AA);
    };
    double fps=t.total_ms>0?1000.0/t.total_ms:0;
    std::ostringstream ss; ss<<std::fixed<<std::setprecision(1);
    ss.str(""); ss<<"FPS: "<<fps;                      put(ss.str(),0);
    ss.str(""); ss<<"Detect: "<<t.detect_ms<<"ms";    put(ss.str(),1);
    ss.str(""); ss<<"Depth:  "<<t.depth_ms<<"ms";     put(ss.str(),2);
    ss.str(""); ss<<"Track:  "<<t.track_ms<<"ms";     put(ss.str(),3);
    ss.str(""); ss<<"Tracks: "<<num_tracks;            put(ss.str(),4);
}

void Renderer::draw_depth_overlay(cv::Mat& frame, const cv::Mat& depth_coloured, float alpha) {
    if(depth_coloured.empty()) return;
    cv::Mat resized;
    cv::resize(depth_coloured,resized,frame.size());
    cv::addWeighted(frame,1.0-alpha,resized,alpha,0,frame);
}

} // namespace pipeline
