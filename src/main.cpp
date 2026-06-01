#include <iostream>
#include "pipeline/threaded_pipeline.hpp"

int main(int argc, char** argv) {
    std::cout << "=== Perception Pipeline (Threaded) ===\n"
              << "  q/ESC: quit | d: depth | e: edges | h: HOG\n"
              << "  c: corners | p: pose | s: save\n\n";

    pipeline::ThreadedPipeline::Config cfg;
    if (argc > 1) cfg.model_path = argv[1];
    if (argc > 2) cfg.camera_id  = std::stoi(argv[2]);

    pipeline::ThreadedPipeline pl(cfg);
    pl.start();
    pl.run_render_loop();   // blocks on main thread (required for GUI)

    return 0;
}
