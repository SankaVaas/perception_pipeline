# Real-time Stereo Perception Pipeline

A C++17 computer vision pipeline built as a learning project for AI/Robotics engineering.
Covers classical CV, stereo depth, YOLOv8 object detection, Kalman filter tracking, and 3D fusion —
all in C++ with OpenCV and ONNX Runtime.

## Concepts covered

| Module | Concepts |
|--------|----------|
| Calibration | Camera intrinsics, lens distortion, stereo rectification, epipolar geometry |
| Preprocess | Gaussian blur, Canny edges, Harris corners, HOG descriptors, letterboxing |
| Depth | SGBM stereo matching, disparity maps, 3D reprojection (Q matrix) |
| Detector | CNN inference, ONNX Runtime C++ API, YOLOv8 output parsing, NMS |
| Tracker | Kalman filter (constant-velocity), IoU matching, track lifecycle |
| Fusion | 2D bbox → 3D world position, depth sampling |
| Systems | C++17, CMake, RAII, chrono profiling, real-time constraints |

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| MSYS2 (Windows) | latest | msys2.org |
| GCC | 16+ | `pacman -S mingw-w64-ucrt-x86_64-gcc` |
| CMake | 3.18+ | `pacman -S mingw-w64-ucrt-x86_64-cmake` |
| OpenCV | 4.13+ | `pacman -S mingw-w64-ucrt-x86_64-opencv` |
| Ninja | any | `pacman -S mingw-w64-ucrt-x86_64-ninja` |
| ONNX Runtime | 1.19+ | See below |

### Windows PATH (required)
Add `C:\msys64\ucrt64\bin` to your Windows system PATH so DLLs are found at runtime.

### ONNX Runtime (Windows)
```bash
cd /d
mkdir onnxruntime && cd onnxruntime
curl -L "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip" -o ort.zip
unzip ort.zip
```

### YOLOv8 model
```bash
# Option A: Python export
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
cp yolov8n.onnx models/

# Option B: Direct download
curl -L "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx" \
     -o models/yolov8n.onnx
```

## Build

```bash
# Configure (point to your ONNX Runtime path)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DONNXRUNTIME_ROOT=/d/onnxruntime/onnxruntime-win-x64-1.19.2

# Build
cmake --build build
```

## Run

From Windows Command Prompt (not MSYS2):
```cmd
D:\repo\perception_pipeline\build\perception_pipeline.exe
```

With a video file:
```cmd
D:\repo\perception_pipeline\build\perception_pipeline.exe models/yolov8n.onnx 0 data/sample.mp4
```

## Key bindings

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `d` | Toggle depth overlay |
| `e` | Toggle Canny edge overlay |
| `h` | Toggle HOG visualisation |
| `c` | Toggle Harris corner detection |
| `s` | Save current frame as JPEG |

## Project structure

```
perception_pipeline/
├── CMakeLists.txt
├── vcpkg.json
├── include/
│   └── pipeline.hpp          ← shared types (Detection, Track, Timings...)
├── models/
│   └── yolov8n.onnx          ← place model here (not tracked in git)
├── data/
│   └── stereo_calib.yaml     ← optional real stereo calibration
└── src/
    ├── main.cpp              ← pipeline entry point
    ├── calibration/          ← Module 1: camera calibration & rectification
    ├── preprocess/           ← Module 2: Gaussian blur, Canny, Harris, HOG
    ├── depth/                ← Module 3: SGBM stereo depth estimation
    ├── detector/             ← Module 4: YOLOv8 via ONNX Runtime
    ├── tracker/              ← Module 5: Kalman filter SORT-style tracker
    ├── fusion/               ← Module 6: 2D bbox → 3D world position
    └── renderer/             ← Module 7: HUD, overlays, profiler
```

## Roadmap

- [x] Project skeleton + CMake + VSCode config
- [x] Camera calibration module (dummy + YAML loader)
- [x] Classical CV: Gaussian, Canny, Harris, HOG
- [x] Stereo depth: SGBM disparity + 3D reprojection
- [x] Stub detector (centred bbox for pipeline testing)
- [x] Kalman filter multi-object tracker
- [x] 2D→3D fusion
- [x] Renderer + FPS profiler
- [ ] YOLOv8 real inference via ONNX Runtime
- [ ] KITTI stereo dataset integration
- [ ] Optical flow (Lucas-Kanade)
- [ ] TensorRT acceleration (future)
- [ ] ROS2 node wrapper (future)

## Stereo calibration

Without a real calibration file the pipeline uses a dummy (depth values approximate).
To use real calibration, run OpenCV's stereo calibration tool and save to `data/stereo_calib.yaml`.
The YAML keys expected are: `left_camera_matrix`, `left_dist_coeffs`, `right_camera_matrix`,
`right_dist_coeffs`, `R`, `T`, `R1`, `R2`, `P1`, `P2`, `Q`, `image_width`, `image_height`.