#include <iostream>
#include <sstream>
#include <iomanip>
#include "pipeline.hpp"
#include "calibration/calibration.hpp"
#include "preprocess/preprocess.hpp"
#include "depth/depth_estimator.hpp"
#include "detector/detector.hpp"
#include "tracker/kalman_tracker.hpp"
#include "fusion/fusion.hpp"
#include "renderer/renderer.hpp"
#include "optical_flow/optical_flow.hpp"
#include "pose/pose_estimator.hpp"

int main(int argc, char** argv) {
    std::cout << "=== Perception Pipeline ===\n"
              << "  q/ESC: quit | d: depth | e: edges | h: HOG\n"
              << "  c: corners | f: optical flow | p: pose | s: save\n\n";

    pipeline::PipelineConfig cfg;
    if (argc > 1) cfg.model_path = argv[1];

    cv::VideoCapture cap;
    if (argc > 3) cap.open(argv[3]);
    if (!cap.isOpened()) cap.open(cfg.camera_id);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return 1; }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.frame_height);
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Input: " << W << "x" << H << "\n";

    // Calibration
    pipeline::StereoCalibration calib =
        pipeline::CalibrationModule::make_dummy(cv::Size(W, H));
    pipeline::CalibrationModule cal_mod;
    cal_mod.load("data/stereo_calib.yaml", calib);
    cv::Mat map1L, map2L, map1R, map2R;
    cal_mod.build_rectify_maps(calib, map1L, map2L, map1R, map2R);

    // Modules
    pipeline::DepthEstimator   depth_est;  depth_est.init();
    pipeline::Detector         detector;   detector.load(cfg.model_path);
    pipeline::KalmanTracker    tracker(cfg.max_age, cfg.min_hits);
    pipeline::FusionModule     fusion;
    pipeline::PreprocessModule prep;
    pipeline::Renderer         renderer;

    pipeline::OpticalFlowModule::Config flow_cfg;
    pipeline::OpticalFlowModule optical_flow(flow_cfg);

    pipeline::PoseEstimator pose_estimator(calib.left);

    // Toggles
    bool show_depth   = true;
    bool show_edges   = false;
    bool show_hog     = false;
    bool show_corners = false;
    bool show_flow    = false;
    bool show_pose    = false;

    cv::Mat frame, depth_vis;
    pipeline::FrameTimings timings;
    pipeline::Stopwatch sw, total_sw;

    std::cout << "Pipeline running. Press 'q' to quit.\n";

    while (true) {
        total_sw.start();
        if (!cap.read(frame) || frame.empty()) break;
        cv::Mat display = frame.clone();

        cv::Mat left_rect  = cal_mod.rectify(frame, map1L, map2L);
        cv::Mat right_rect = left_rect;

        // Classical CV
        sw.start();
        if (show_edges) {
            cv::Mat e = prep.canny_edges(frame), eb;
            cv::cvtColor(e, eb, cv::COLOR_GRAY2BGR);
            cv::addWeighted(display, 0.7, eb, 0.3, 0, display);
        }
        if (show_corners) { cv::Mat cr; display = prep.harris_corners(display, cr); }
        if (show_hog) {
            cv::Mat hv = prep.hog_visualise(frame), hr;
            cv::resize(hv, hr, cv::Size(display.cols/4, display.rows/4));
            hr.copyTo(display(cv::Rect(0, 0, hr.cols, hr.rows)));
        }
        timings.preprocess_ms = sw.stop_ms();

        // Optical flow
        if (show_flow) {
            cv::Mat flow_vis = optical_flow.process(frame);
            cv::addWeighted(display, 0.5, flow_vis, 0.5, 0, display);
            cv::Point2f ego = optical_flow.estimate_ego_motion();
            std::ostringstream ess;
            ess << std::fixed << std::setprecision(1)
                << "Ego: (" << ego.x << ", " << ego.y << ") px/f";
            cv::putText(display, ess.str(),
                        cv::Point(10, display.rows-20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(200,200,0), 1, cv::LINE_AA);
        } else {
            optical_flow.reset();
        }

        // Depth
        sw.start();
        cv::Mat disp  = depth_est.compute_disparity(left_rect, right_rect);
        cv::Mat cloud = depth_est.disparity_to_3d(disp, calib.Q);
        depth_vis     = depth_est.colourize_disparity(disp);
        timings.depth_ms = sw.stop_ms();

        // Detection
        sw.start();
        auto dets = detector.detect(frame, cfg.conf_threshold,
                                    cfg.nms_threshold, cfg.input_size);
        timings.detect_ms = sw.stop_ms();

        fusion.fuse(dets, cloud);

        // Tracking
        sw.start();
        auto tracks = tracker.update(dets);
        fusion.fuse_tracks(tracks, cloud);
        timings.track_ms = sw.stop_ms();

        // Pose estimation
        if (show_pose) {
            for (const auto& d : dets) {
                if (d.bbox.area() < 400) continue; // skip tiny boxes
                auto obj_size = pipeline::PoseEstimator::default_size(d.class_id);
                auto pose = pose_estimator.estimate(d.bbox, obj_size);
                pose_estimator.draw_axes(display, pose, d.bbox, 0.15f);
                pose_estimator.draw_info(display, pose, d.bbox);
            }
        }

        // Render
        if (show_depth && !depth_vis.empty())
            renderer.draw_depth_overlay(display, depth_vis, 0.35f);
        renderer.draw_tracks(display, tracks);
        renderer.draw_detections(display, dets);
        renderer.draw_hud(display, timings, (int)tracks.size());
        timings.total_ms = total_sw.stop_ms();

        cv::imshow("Perception Pipeline", display);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == 'd') show_depth   = !show_depth;
        if (key == 'e') show_edges   = !show_edges;
        if (key == 'h') show_hog     = !show_hog;
        if (key == 'c') show_corners = !show_corners;
        if (key == 'f') { show_flow  = !show_flow; optical_flow.reset(); }
        if (key == 'p') show_pose    = !show_pose;
        if (key == 's') {
            cv::imwrite("frame_saved.jpg", display);
            std::cout << "Saved.\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
