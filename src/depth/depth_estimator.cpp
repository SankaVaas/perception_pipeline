#include "depth_estimator.hpp"

namespace pipeline {

void DepthEstimator::init(int min_disp, int num_disp, int block_size) {
    int cn = 1;
    sgbm_ = cv::StereoSGBM::create(min_disp, num_disp, block_size,
        8*cn*block_size*block_size, 32*cn*block_size*block_size,
        1, 0, 10, 100, 32, cv::StereoSGBM::MODE_SGBM_3WAY);
}

cv::Mat DepthEstimator::compute_disparity(const cv::Mat& left_rect, const cv::Mat& right_rect) {
    cv::Mat lg, rg, disp16;
    if (left_rect.channels()==3) { cv::cvtColor(left_rect,lg,cv::COLOR_BGR2GRAY); cv::cvtColor(right_rect,rg,cv::COLOR_BGR2GRAY); }
    else { lg=left_rect; rg=right_rect; }
    sgbm_->compute(lg, rg, disp16);
    cv::Mat disparity;
    disp16.convertTo(disparity, CV_32F, 1.0/16.0);
    return disparity;
}

cv::Mat DepthEstimator::disparity_to_3d(const cv::Mat& disparity, const cv::Mat& Q) {
    cv::Mat points3d;
    cv::reprojectImageTo3D(disparity, points3d, Q, true);
    return points3d;
}

float DepthEstimator::sample_depth_at_bbox(const cv::Mat& depth_map, const cv::Rect& bbox) {
    if (depth_map.empty()) return -1.f;
    cv::Rect roi = bbox & cv::Rect(0,0,depth_map.cols,depth_map.rows);
    if (roi.empty()) return -1.f;
    int pw = std::max(1, roi.width/4), ph = std::max(1, roi.height/4);
    cv::Rect patch(roi.x+roi.width/2-pw/2, roi.y+roi.height/2-ph/2, pw, ph);
    patch &= cv::Rect(0,0,depth_map.cols,depth_map.rows);
    std::vector<float> zvals;
    cv::Mat sub = depth_map(patch);
    for (int r=0;r<sub.rows;r++)
        for (int c=0;c<sub.cols;c++) {
            float z = sub.at<cv::Vec3f>(r,c)[2];
            if (z>0.01f && z<100.f) zvals.push_back(z);
        }
    if (zvals.empty()) return -1.f;
    std::nth_element(zvals.begin(), zvals.begin()+zvals.size()/2, zvals.end());
    return zvals[zvals.size()/2];
}

cv::Mat DepthEstimator::colourize_disparity(const cv::Mat& disparity) {
    cv::Mat vis;
    double mn, mx;
    cv::minMaxLoc(disparity, &mn, &mx);
    disparity.convertTo(vis, CV_8U, 255.0/(mx-mn+1e-5), -mn*255.0/(mx-mn+1e-5));
    cv::applyColorMap(vis, vis, cv::COLORMAP_TURBO);
    return vis;
}

} // namespace pipeline
