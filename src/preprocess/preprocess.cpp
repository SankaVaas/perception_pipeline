#include "preprocess.hpp"

namespace pipeline {

cv::Mat PreprocessModule::gaussian_blur(const cv::Mat& src, int kernel_size, double sigma) {
    cv::Mat out;
    cv::GaussianBlur(src, out, cv::Size(kernel_size, kernel_size), sigma);
    return out;
}

cv::Mat PreprocessModule::canny_edges(const cv::Mat& src, double low, double high) {
    cv::Mat gray, blurred, edges;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 1.4);
    cv::Canny(blurred, edges, low, high);
    return edges;
}

cv::Mat PreprocessModule::harris_corners(const cv::Mat& src, cv::Mat& corner_response,
                                         int block_size, int ksize, double k, double thresh) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);
    cv::cornerHarris(gray_f, corner_response, block_size, ksize, k);
    cv::Mat norm;
    cv::normalize(corner_response, norm, 0, 1, cv::NORM_MINMAX);
    cv::Mat result;
    src.copyTo(result);
    for (int r = 0; r < norm.rows; r++)
        for (int c = 0; c < norm.cols; c++)
            if (norm.at<float>(r,c) > thresh)
                cv::circle(result, cv::Point(c,r), 3, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    return result;
}

cv::Mat PreprocessModule::hog_visualise(const cv::Mat& src) {
    cv::Mat gray, roi;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;
    cv::resize(gray, roi, cv::Size(64, 128));
    cv::Mat gx, gy, mag, angle;
    cv::Sobel(roi, gx, CV_32F, 1, 0, 1);
    cv::Sobel(roi, gy, CV_32F, 0, 1, 1);
    cv::cartToPolar(gx, gy, mag, angle);
    cv::Mat vis;
    cv::normalize(mag, vis, 0, 255, cv::NORM_MINMAX);
    vis.convertTo(vis, CV_8U);
    cv::applyColorMap(vis, vis, cv::COLORMAP_MAGMA);
    return vis;
}

cv::Mat PreprocessModule::letterbox(const cv::Mat& src, int target_size,
                                    float& scale_x, float& scale_y,
                                    int& pad_x, int& pad_y) {
    float scale = std::min((float)target_size / src.cols,
                           (float)target_size / src.rows);
    scale_x = scale_y = scale;
    int new_w = (int)(src.cols * scale);
    int new_h = (int)(src.rows * scale);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    pad_x = (target_size - new_w) / 2;
    pad_y = (target_size - new_h) / 2;
    cv::Mat canvas(target_size, target_size, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);
    return cv::dnn::blobFromImage(rgb);
}

} // namespace pipeline
