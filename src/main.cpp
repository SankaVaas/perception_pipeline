#include <iostream>
#include "pipeline.hpp"
#include "calibration/calibration.hpp"
#include "preprocess/preprocess.hpp"
#include "depth/depth_estimator.hpp"
#include "detector/detector.hpp"
#include "tracker/kalman_tracker.hpp"
#include "fusion/fusion.hpp"
#include "renderer/renderer.hpp"

int main(int argc, char** argv) {
    std::cout << "=== Perception Pipeline ===\n"
              << "  q/ESC: quit | d: depth | e: edges | h: HOG | c: corners | s: save\n\n";

    pipeline::PipelineConfig cfg;
    if(argc>1) cfg.model_path = argv[1];

    cv::VideoCapture cap;
    if(argc>3) cap.open(argv[3]);
    if(!cap.isOpened()) cap.open(cfg.camera_id);
    if(!cap.isOpened()) { std::cerr<<"Cannot open camera\n"; return 1; }

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.frame_height);
    const int W=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout<<"Input: "<<W<<"x"<<H<<"\n";

    pipeline::StereoCalibration calib =
        pipeline::CalibrationModule::make_dummy(cv::Size(W,H));
    pipeline::CalibrationModule cal_mod;
    cal_mod.load("data/stereo_calib.yaml", calib);
    cv::Mat map1L,map2L,map1R,map2R;
    cal_mod.build_rectify_maps(calib,map1L,map2L,map1R,map2R);

    pipeline::DepthEstimator  depth_est; depth_est.init();
    pipeline::Detector        detector;  detector.load(cfg.model_path);
    pipeline::KalmanTracker   tracker(cfg.max_age, cfg.min_hits);
    pipeline::FusionModule    fusion;
    pipeline::PreprocessModule prep;
    pipeline::Renderer        renderer;

    bool show_depth=true, show_edges=false, show_hog=false, show_corners=false;
    cv::Mat frame, depth_vis;
    pipeline::FrameTimings timings;
    pipeline::Stopwatch sw, total_sw;

    std::cout<<"Pipeline running. Press 'q' to quit.\n";

    while(true) {
        total_sw.start();
        if(!cap.read(frame)||frame.empty()) break;
        cv::Mat display=frame.clone();

        cv::Mat left_rect = cal_mod.rectify(frame,map1L,map2L);
        cv::Mat right_rect = left_rect;

        sw.start();
        if(show_edges) {
            cv::Mat e=prep.canny_edges(frame), eb;
            cv::cvtColor(e,eb,cv::COLOR_GRAY2BGR);
            cv::addWeighted(display,0.7,eb,0.3,0,display);
        }
        if(show_corners) { cv::Mat cr; display=prep.harris_corners(display,cr); }
        if(show_hog) {
            cv::Mat hv=prep.hog_visualise(frame), hr;
            cv::resize(hv,hr,cv::Size(display.cols/4,display.rows/4));
            hr.copyTo(display(cv::Rect(0,0,hr.cols,hr.rows)));
        }
        timings.preprocess_ms=sw.stop_ms();

        sw.start();
        cv::Mat disp=depth_est.compute_disparity(left_rect,right_rect);
        cv::Mat cloud=depth_est.disparity_to_3d(disp,calib.Q);
        depth_vis=depth_est.colourize_disparity(disp);
        timings.depth_ms=sw.stop_ms();

        sw.start();
        auto dets=detector.detect(frame,cfg.conf_threshold,cfg.nms_threshold,cfg.input_size);
        timings.detect_ms=sw.stop_ms();

        fusion.fuse(dets,cloud);

        sw.start();
        auto tracks=tracker.update(dets);
        fusion.fuse_tracks(tracks,cloud);
        timings.track_ms=sw.stop_ms();

        if(show_depth&&!depth_vis.empty()) renderer.draw_depth_overlay(display,depth_vis,0.35f);
        renderer.draw_tracks(display,tracks);
        renderer.draw_detections(display,dets);
        renderer.draw_hud(display,timings,(int)tracks.size());
        timings.total_ms=total_sw.stop_ms();

        cv::imshow("Perception Pipeline",display);
        int key=cv::waitKey(1)&0xFF;
        if(key=='q'||key==27) break;
        if(key=='d') show_depth=!show_depth;
        if(key=='e') show_edges=!show_edges;
        if(key=='h') show_hog=!show_hog;
        if(key=='c') show_corners=!show_corners;
        if(key=='s') { cv::imwrite("frame_saved.jpg",display); std::cout<<"Saved.\n"; }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
