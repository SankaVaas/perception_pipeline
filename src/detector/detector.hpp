#pragma once
#include "../../include/pipeline.hpp"

namespace pipeline {

class Detector {
public:
    bool load(const std::string& model_path, bool use_cuda = false);
    std::vector<Detection> detect(const cv::Mat& frame,
                                  float conf_thresh = 0.45f,
                                  float nms_thresh  = 0.50f,
                                  int   input_size  = 640);
    bool is_loaded() const { return loaded_; }
    static const std::vector<std::string>& coco_names();
private:
    bool loaded_ = false;
};

} // namespace pipeline
