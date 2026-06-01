#pragma once
#include "../../include/pipeline.hpp"
#include "../calibration/calibration.hpp"
#include "../depth/depth_estimator.hpp"
#include "../detector/detector.hpp"
#include "../tracker/kalman_tracker.hpp"
#include "../fusion/fusion.hpp"
#include "../renderer/renderer.hpp"
#include "../optical_flow/optical_flow.hpp"
#include "../pose/pose_estimator.hpp"
#include "../preprocess/preprocess.hpp"

#include <thread>
#include <atomic>

namespace pipeline {

class ThreadedPipeline {
public:
    struct Config {
        int         camera_id     = 0;
        int         width         = 640;
        int         height        = 480;
        std::string model_path    = "models/yolov8n.onnx";
        float       conf_thresh   = 0.45f;
        float       nms_thresh    = 0.50f;
        int         input_size    = 640;
        int         max_age       = 10;
        int         min_hits      = 3;
    };

    explicit ThreadedPipeline(const Config& cfg);
    ~ThreadedPipeline();

    void start();
    void stop();
    bool is_running() const { return running_; }

    // Toggle display modes (called from main/render thread)
    void toggle_depth()    { show_depth_    = !show_depth_;    }
    void toggle_edges()    { show_edges_    = !show_edges_;    }
    void toggle_hog()      { show_hog_      = !show_hog_;      }
    void toggle_corners()  { show_corners_  = !show_corners_;  }
    void toggle_flow()     { show_flow_     = !show_flow_;     }
    void toggle_pose()     { show_pose_     = !show_pose_;     }
    void save_frame()      { save_next_     = true;            }

    // Run the render loop on the calling thread (must be main thread for GUI)
    void run_render_loop();

private:
    void capture_thread_fn();
    void inference_thread_fn();

    Config cfg_;

    // Inter-thread buffers
    LatestFrameBuffer<cv::Mat>          capture_buf_;   // raw frames
    LatestFrameBuffer<InferenceResult>  result_buf_;    // processed results

    // Threads
    std::thread capture_thread_;
    std::thread inference_thread_;
    std::atomic<bool> running_{false};

    // Modules (owned by inference thread — no sharing needed)
    std::unique_ptr<DepthEstimator>    depth_est_;
    std::unique_ptr<Detector>          detector_;
    std::unique_ptr<KalmanTracker>     tracker_;
    std::unique_ptr<FusionModule>      fusion_;
    std::unique_ptr<PreprocessModule>  prep_;
    std::unique_ptr<OpticalFlowModule> optical_flow_;

    // Calibration (read-only after init — safe to share)
    StereoCalibration calib_;
    cv::Mat map1L_, map2L_, map1R_, map2R_;

    // Render-thread state
    std::unique_ptr<Renderer>      renderer_;
    std::unique_ptr<PoseEstimator> pose_estimator_;

    // Display toggles (atomic so render thread can read, main thread writes)
    std::atomic<bool> show_depth_   {true};
    std::atomic<bool> show_edges_   {false};
    std::atomic<bool> show_hog_     {false};
    std::atomic<bool> show_corners_ {false};
    std::atomic<bool> show_flow_    {false};
    std::atomic<bool> show_pose_    {false};
    std::atomic<bool> save_next_    {false};

    // FPS tracking
    std::atomic<double> capture_fps_  {0};
    std::atomic<double> inference_fps_{0};
};

} // namespace pipeline
