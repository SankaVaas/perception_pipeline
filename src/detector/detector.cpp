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

bool Detector::load(const std::string& model_path, bool use_cuda) {
#ifndef HAS_ONNX
    std::cerr << "[Detector] ONNX Runtime not compiled in - stub mode.\n";
    loaded_ = false;
    return false;
#else
    session_opts_.SetIntraOpNumThreads(4);
    session_opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    if (use_cuda) {
        OrtCUDAProviderOptions cuda_opts{};
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
    }

    try {
        std::wstring wpath(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_opts_);

        for (size_t i = 0; i < session_->GetInputCount(); i++)
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator_).get());
        for (size_t i = 0; i < session_->GetOutputCount(); i++)
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator_).get());

        // Print output shape for diagnostics
        auto type_info = session_->GetOutputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        std::cout << "[Detector] Output shape: [";
        for (size_t i = 0; i < shape.size(); i++)
            std::cout << shape[i] << (i+1<shape.size()?", ":"");
        std::cout << "]\n";

        loaded_ = true;
        std::cout << "[Detector] Loaded: " << model_path << "\n";
        std::cout << "[Detector] Input:  " << input_names_[0] << "\n";
        std::cout << "[Detector] Output: " << output_names_[0] << "\n";
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[Detector] ORT error: " << e.what() << "\n";
        return false;
    }
#endif
}

std::vector<Detection> Detector::detect(const cv::Mat& frame,
                                        float conf_thresh, float nms_thresh,
                                        int input_size) {
    if (frame.empty()) return {};

#ifndef HAS_ONNX
    Detection d;
    d.class_id = 0; d.confidence = 0.9f;
    d.bbox = cv::Rect(frame.cols/4, frame.rows/4, frame.cols/2, frame.rows/2);
    return {d};
#else
    if (!loaded_) {
        Detection d;
        d.class_id = 0; d.confidence = 0.9f;
        d.bbox = cv::Rect(frame.cols/4, frame.rows/4, frame.cols/2, frame.rows/2);
        return {d};
    }

    float scale_x, scale_y;
    int   pad_x, pad_y;
    PreprocessModule prep;
    cv::Mat blob = prep.letterbox(frame, input_size, scale_x, scale_y, pad_x, pad_y);

    std::array<int64_t,4> input_shape{1, 3, input_size, input_size};
    size_t n_elems = 3 * input_size * input_size;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(blob.data),
        n_elems, input_shape.data(), input_shape.size());

    const char* in_name  = input_names_[0].c_str();
    const char* out_name = output_names_[0].c_str();

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        &in_name, &input_tensor, 1,
        &out_name, 1);

    auto shape      = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* data = outputs[0].GetTensorMutableData<float>();

    // ── Handle both output layouts ────────────────────────────────────────
    // Standard YOLOv8 export: [1, 84, 8400] — cols = proposals
    // This model:             [1, 8400, 84] — rows = proposals (transposed)
    bool transposed = (shape[1] > shape[2]); // if dim1 > dim2, it's [1,8400,84]
    int num_proposals = transposed ? (int)shape[1] : (int)shape[2];
    int num_fields    = transposed ? (int)shape[2] : (int)shape[1];
    int num_classes   = num_fields - 4;

    return parse_output(data, num_proposals, num_classes, transposed,
                        conf_thresh, nms_thresh,
                        scale_x, scale_y, pad_x, pad_y,
                        frame.cols, frame.rows);
#endif
}

std::vector<Detection> Detector::parse_output(const float* data,
                                              int num_props, int num_classes,
                                              bool transposed,
                                              float conf_thresh, float nms_thresh,
                                              float scale_x, float scale_y,
                                              int pad_x, int pad_y,
                                              int orig_w, int orig_h) {
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;

    for (int i = 0; i < num_props; i++) {
        float cx, cy, w, h;
        int   best_cls   = -1;
        float best_score = conf_thresh;

        if (transposed) {
            // Layout [8400, 84]: row i = {cx,cy,w,h, s0..s79}
            const float* row = data + i * (4 + num_classes);
            cx = row[0]; cy = row[1]; w = row[2]; h = row[3];
            for (int c = 0; c < num_classes; c++) {
                if (row[4+c] > best_score) { best_score = row[4+c]; best_cls = c; }
            }
        } else {
            // Layout [84, 8400]: col i
            cx = data[0*num_props+i]; cy = data[1*num_props+i];
            w  = data[2*num_props+i]; h  = data[3*num_props+i];
            for (int c = 0; c < num_classes; c++) {
                float s = data[(4+c)*num_props+i];
                if (s > best_score) { best_score = s; best_cls = c; }
            }
        }

        if (best_cls < 0) continue;

        float x1 = (cx - w/2 - pad_x) / scale_x;
        float y1 = (cy - h/2 - pad_y) / scale_y;
        float bw = w / scale_x;
        float bh = h / scale_y;

        x1 = std::max(0.f, std::min(x1, (float)orig_w));
        y1 = std::max(0.f, std::min(y1, (float)orig_h));
        bw = std::min(bw, (float)orig_w - x1);
        bh = std::min(bh, (float)orig_h - y1);

        boxes.emplace_back((int)x1,(int)y1,(int)bw,(int)bh);
        scores.push_back(best_score);
        class_ids.push_back(best_cls);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, kept);

    std::vector<Detection> dets;
    for (int idx : kept) {
        Detection d;
        d.class_id   = class_ids[idx];
        d.confidence = scores[idx];
        d.bbox       = boxes[idx];
        dets.push_back(d);
    }
    return dets;
}

} // namespace pipeline
