#include "threaded_pipeline.hpp"
#include <iostream>
#include <chrono>

namespace pipeline {

ThreadedPipeline::ThreadedPipeline(const Config& cfg) : cfg_(cfg) {
    // ── Init calibration (shared read-only data) ──────────────────────────
    CalibrationModule cal_mod;
    calib_ = CalibrationModule::make_dummy(cv::Size(cfg_.width, cfg_.height));
    cal_mod.load("data/stereo_calib.yaml", calib_);
    cal_mod.build_rectify_maps(calib_, map1L_, map2L_, map1R_, map2R_);

    // ── Init inference-thread modules ─────────────────────────────────────
    depth_est_    = std::make_unique<DepthEstimator>();
    detector_     = std::make_unique<Detector>();
    tracker_      = std::make_unique<KalmanTracker>(cfg_.max_age, cfg_.min_hits);
    fusion_       = std::make_unique<FusionModule>();
    prep_         = std::make_unique<PreprocessModule>();
    optical_flow_ = std::make_unique<OpticalFlowModule>();

    depth_est_->init();
    detector_->load(cfg_.model_path);

    // ── Init render-thread modules ────────────────────────────────────────
    renderer_       = std::make_unique<Renderer>();
    pose_estimator_ = std::make_unique<PoseEstimator>(calib_.left);
}

ThreadedPipeline::~ThreadedPipeline() { stop(); }

void ThreadedPipeline::start() {
    running_ = true;
    capture_thread_   = std::thread(&ThreadedPipeline::capture_thread_fn,   this);
    inference_thread_ = std::thread(&ThreadedPipeline::inference_thread_fn, this);
}

void ThreadedPipeline::stop() {
    running_ = false;
    capture_buf_.stop();
    result_buf_.stop();
    if (capture_thread_.joinable())   capture_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread 1: Capture
// Runs as fast as the camera allows. Pushes latest frame to capture_buf_.
// If inference is slower than capture, frames are dropped (latest wins).
// ─────────────────────────────────────────────────────────────────────────────
void ThreadedPipeline::capture_thread_fn() {
    cv::VideoCapture cap(cfg_.camera_id);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg_.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.height);

    if (!cap.isOpened()) {
        std::cerr << "[Capture] Cannot open camera " << cfg_.camera_id << "\n";
        running_ = false;
        capture_buf_.stop();
        return;
    }
    std::cout << "[Capture] Thread started\n";

    // FPS measurement
    int frame_count = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (running_) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) continue;
        capture_buf_.push(frame);

        // FPS tracking
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (elapsed >= 1.0) {
            capture_fps_ = frame_count / elapsed;
            frame_count  = 0;
            t_start      = now;
        }
    }
    cap.release();
    capture_buf_.stop();
    std::cout << "[Capture] Thread stopped\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread 2: Inference
// Pops latest frame, runs the full CV/DNN pipeline, pushes result.
// ─────────────────────────────────────────────────────────────────────────────
void ThreadedPipeline::inference_thread_fn() {
    std::cout << "[Inference] Thread started\n";

    int frame_count = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    Stopwatch sw, total_sw;

    while (running_) {
        cv::Mat frame;
        if (!capture_buf_.pop(frame, 100)) continue;

        InferenceResult result;
        result.frame = frame.clone();
        total_sw.start();

        // Rectify
        CalibrationModule cal;
        cv::Mat left_rect  = cal.rectify(frame, map1L_, map2L_);
        cv::Mat right_rect = left_rect;

        // Depth
        sw.start();
        cv::Mat disp  = depth_est_->compute_disparity(left_rect, right_rect);
        cv::Mat cloud = depth_est_->disparity_to_3d(disp, calib_.Q);
        result.depth_vis = depth_est_->colourize_disparity(disp);
        result.timings.depth_ms = sw.stop_ms();

        // Detection
        sw.start();
        result.detections = detector_->detect(frame,
            cfg_.conf_thresh, cfg_.nms_thresh, cfg_.input_size);
        result.timings.detect_ms = sw.stop_ms();

        // Fusion
        fusion_->fuse(result.detections, cloud);

        // Tracking
        sw.start();
        result.tracks = tracker_->update(result.detections);
        fusion_->fuse_tracks(result.tracks, cloud);
        result.timings.track_ms = sw.stop_ms();

        result.timings.total_ms = total_sw.stop_ms();
        result_buf_.push(std::move(result));

        // FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (elapsed >= 1.0) {
            inference_fps_ = frame_count / elapsed;
            frame_count    = 0;
            t_start        = now;
        }
    }
    result_buf_.stop();
    std::cout << "[Inference] Thread stopped\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread 3: Render (runs on main thread — required by OpenCV/Qt windowing)
// ─────────────────────────────────────────────────────────────────────────────
void ThreadedPipeline::run_render_loop() {
    std::cout << "[Render] Loop started\n";
    PreprocessModule prep;

    while (running_) {
        InferenceResult result;
        if (!result_buf_.pop(result, 100)) continue;

        cv::Mat display = result.frame.clone();

        // Classical CV overlays (rendered on the display copy)
        if (show_edges_) {
            cv::Mat e = prep.canny_edges(result.frame), eb;
            cv::cvtColor(e, eb, cv::COLOR_GRAY2BGR);
            cv::addWeighted(display, 0.7, eb, 0.3, 0, display);
        }
        if (show_corners_) {
            cv::Mat cr;
            display = prep.harris_corners(display, cr);
        }
        if (show_hog_) {
            cv::Mat hv = prep.hog_visualise(result.frame), hr;
            cv::resize(hv, hr, cv::Size(display.cols/4, display.rows/4));
            hr.copyTo(display(cv::Rect(0,0,hr.cols,hr.rows)));
        }

        // Depth overlay
        if (show_depth_ && !result.depth_vis.empty())
            renderer_->draw_depth_overlay(display, result.depth_vis, 0.35f);

        // Pose
        if (show_pose_) {
            for (const auto& d : result.detections) {
                if (d.bbox.area() < 400) continue;
                float sz = d.position_3d.z;
                if (sz > 0.05f) {
                    auto fp = pose_estimator_->estimate_fused(d.bbox, sz);
                    pose_estimator_->draw_fused_axes(display, fp, 0.15f);
                    pose_estimator_->draw_fused_info(display, fp, d.bbox);
                } else {
                    auto obj = PoseEstimator::default_size(d.class_id);
                    auto p   = pose_estimator_->estimate(d.bbox, obj);
                    pose_estimator_->draw_axes(display, p, d.bbox, 0.15f);
                    pose_estimator_->draw_info(display, p, d.bbox);
                }
            }
        }

        renderer_->draw_tracks(display, result.tracks);
        renderer_->draw_detections(display, result.detections);
        renderer_->draw_hud(display, result.timings,
                            (int)result.tracks.size());

        // Thread FPS overlay
        std::ostringstream fps_ss;
        fps_ss << std::fixed << std::setprecision(1)
               << "Cap:" << capture_fps_.load()
               << " Inf:" << inference_fps_.load() << " fps";
        cv::putText(display, fps_ss.str(),
                    cv::Point(10, display.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(180,255,180), 1, cv::LINE_AA);

        cv::imshow("Perception Pipeline [Threaded]", display);

        if (save_next_.exchange(false))
            cv::imwrite("frame_saved.jpg", display);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)  { stop(); break; }
        if (key == 'd') toggle_depth();
        if (key == 'e') toggle_edges();
        if (key == 'h') toggle_hog();
        if (key == 'c') toggle_corners();
        if (key == 'p') toggle_pose();
        if (key == 's') save_frame();
    }

    cv::destroyAllWindows();
    std::cout << "[Render] Loop stopped\n";
}

} // namespace pipeline
