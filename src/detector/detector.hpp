#pragma once
#include "../../include/pipeline.hpp"

#ifdef HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

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
    std::vector<Detection> parse_output(const float* data,
                                        int num_proposals, int num_classes,
                                        bool transposed,
                                        float conf_thresh, float nms_thresh,
                                        float scale_x, float scale_y,
                                        int pad_x, int pad_y,
                                        int orig_w, int orig_h);
    bool loaded_ = false;

#ifdef HAS_ONNX
    Ort::Env                         env_{ORT_LOGGING_LEVEL_WARNING, "detector"};
    Ort::SessionOptions              session_opts_;
    std::unique_ptr<Ort::Session>    session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string>         input_names_;
    std::vector<std::string>         output_names_;
#endif
};

} // namespace pipeline
