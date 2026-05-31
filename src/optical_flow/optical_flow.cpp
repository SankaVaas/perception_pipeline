#include "optical_flow.hpp"

namespace pipeline {

void OpticalFlowModule::reset() {
    prev_gray_.release();
    prev_pts_.clear(); curr_pts_.clear(); flow_.clear();
    initialized_ = false; frame_count_ = 0;
}

void OpticalFlowModule::detect_features(const cv::Mat& gray) {
    cv::goodFeaturesToTrack(gray, prev_pts_, cfg_.max_corners,
        cfg_.quality_level, cfg_.min_distance,
        cv::Mat(), cfg_.block_size, false, 0.04);
    if (!prev_pts_.empty())
        cv::cornerSubPix(gray, prev_pts_, cv::Size(5,5), cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,
                             cfg_.max_iter, cfg_.epsilon));
}

cv::Mat OpticalFlowModule::process(const cv::Mat& frame) {
    cv::Mat result; frame.copyTo(result);
    cv::Mat gray;
    if (frame.channels()==3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();
    frame_count_++;

    if (!initialized_ || prev_pts_.empty()) {
        detect_features(gray);
        gray.copyTo(prev_gray_);
        curr_pts_ = prev_pts_;
        flow_.assign(prev_pts_.size(), cv::Point2f(0,0));
        initialized_ = !prev_pts_.empty();
        return result;
    }

    cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts_, curr_pts_,
        status_, err_, cfg_.win_size, cfg_.max_level,
        cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,
                         cfg_.max_iter, cfg_.epsilon));

    std::vector<cv::Point2f> good_prev, good_curr;
    flow_.clear();
    for (size_t i=0; i<status_.size(); i++) {
        if (!status_[i]) continue;
        good_prev.push_back(prev_pts_[i]);
        good_curr.push_back(curr_pts_[i]);
        flow_.push_back(curr_pts_[i] - prev_pts_[i]);
    }
    curr_pts_ = good_curr;

    if (cfg_.redetect_on_loss &&
        (int)curr_pts_.size() < cfg_.redetect_threshold) {
        detect_features(gray);
        gray.copyTo(prev_gray_);
        flow_.assign(prev_pts_.size(), cv::Point2f(0,0));
        curr_pts_ = prev_pts_;
        return result;
    }

    draw_flow(result);
    gray.copyTo(prev_gray_);
    prev_pts_ = curr_pts_;
    return result;
}

void OpticalFlowModule::draw_flow(cv::Mat& frame) const {
    for (size_t i=0; i<curr_pts_.size(); i++) {
        if (i >= flow_.size()) break;
        float mag = cv::norm(flow_[i]);
        if (mag < cfg_.min_flow_mag) {
            if (cfg_.draw_points)
                cv::circle(frame, curr_pts_[i], 2,
                           cv::Scalar(100,100,100), -1, cv::LINE_AA);
            continue;
        }
        float angle = std::atan2(flow_[i].y, flow_[i].x)*180.f/CV_PI;
        if (angle < 0) angle += 360.f;
        float norm_mag = std::min(mag/20.f, 1.f);
        cv::Mat hsv_px(1,1,CV_8UC3,
            cv::Scalar((int)(angle/2), 255, (int)(norm_mag*255)));
        cv::Mat bgr_px;
        cv::cvtColor(hsv_px, bgr_px, cv::COLOR_HSV2BGR);
        cv::Vec3b col = bgr_px.at<cv::Vec3b>(0,0);
        cv::Scalar colour(col[0], col[1], col[2]);
        if (cfg_.draw_vectors) {
            cv::Point2f end = curr_pts_[i] + flow_[i]*3.f;
            cv::arrowedLine(frame, curr_pts_[i], end,
                            colour, 1, cv::LINE_AA, 0, 0.3);
        }
        if (cfg_.draw_points)
            cv::circle(frame, curr_pts_[i], 3, colour, -1, cv::LINE_AA);
    }
}

cv::Point2f OpticalFlowModule::estimate_ego_motion() const {
    if (flow_.empty()) return {0,0};
    std::vector<float> dx, dy;
    for (const auto& f:flow_) { dx.push_back(f.x); dy.push_back(f.y); }
    std::nth_element(dx.begin(), dx.begin()+dx.size()/2, dx.end());
    std::nth_element(dy.begin(), dy.begin()+dy.size()/2, dy.end());
    return {dx[dx.size()/2], dy[dy.size()/2]};
}

} // namespace pipeline
