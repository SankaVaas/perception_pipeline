#include "detector.hpp"
#include "../preprocess/preprocess.hpp"
#include <iostream>

namespace pipeline {

const std::vector<std::string>& Detector::coco_names() {
    static const std::vector<std::string> names = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    };
    return names;
}

bool Detector::load(const std::string& model_path, bool /*use_cuda*/) {
    std::cout << "[Detector] ONNX Runtime not compiled - stub mode. Model: " << model_path << "\n";
    loaded_ = false;
    return false;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame,
                                        float /*conf*/, float /*nms*/, int /*sz*/) {
    if (frame.empty()) return {};
    // Stub: one fake centred detection so the pipeline can be tested end-to-end
    Detection d;
    d.class_id   = 0;
    d.confidence = 0.9f;
    d.bbox = cv::Rect(frame.cols/4, frame.rows/4, frame.cols/2, frame.rows/2);
    return {d};
}

} // namespace pipeline
