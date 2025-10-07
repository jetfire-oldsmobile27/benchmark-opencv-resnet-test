// example usage: .\build\build\Release\resnet-benchmark  .\models\resnet50_emb.onnx --mode full --input-w 128 --input-h 128 --frames 50 --warmup 10 --iter 500 --threads 0 --gpu false
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using ms_d = std::chrono::duration<double, std::milli>;

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <model.onnx> [options]\n\n"
              << "Options:\n"
              << "  --mode full|infer       (default: full)\n"
              << "  --input-w N             (default: 128)\n"
              << "  --input-h N             (default: 128)\n"
              << "  --src-w N               (default: same as input)\n"
              << "  --src-h N               (default: same as input)\n"
              << "  --frames N              (default: 50)\n"
              << "  --warmup N              (default: 10)\n"
              << "  --iter N                (default: 500)\n"
              << "  --threads N             (OpenCV threads; 0 = auto, default: 0)\n"
              << "  --rand                  (fill frames with random noise)\n"
              << "  --gpu true|false        (use CUDA backend if true, default: false)\n"
              << "  --mean r g b            (default 0.485 0.456 0.406)\n              --std r g b             (default 0.229 0.224 0.225)\n";
}

static double median_of_vector(std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    };
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    double med = v[v.size()/2];
    if (v.size() % 2 == 0) {
        auto it = std::max_element(v.begin(), v.begin() + v.size()/2);
        med = (med + *it) * 0.5;
    }
    return med;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string mode = "full";
    int in_w = 128, in_h = 128;
    int src_w = -1, src_h = -1;
    int num_frames = 50;
    int warmup = 10;
    int iterations = 500;
    int threads = 0;
    bool rand_frames = false;
    bool use_gpu = false;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mode" && i+1<argc) { mode = argv[++i]; }
        else if (a == "--input-w" && i+1<argc) { in_w = std::stoi(argv[++i]); }
        else if (a == "--input-h" && i+1<argc) { in_h = std::stoi(argv[++i]); }
        else if (a == "--src-w" && i+1<argc) { src_w = std::stoi(argv[++i]); }
        else if (a == "--src-h" && i+1<argc) { src_h = std::stoi(argv[++i]); }
        else if (a == "--frames" && i+1<argc) { num_frames = std::stoi(argv[++i]); }
        else if (a == "--warmup" && i+1<argc) { warmup = std::stoi(argv[++i]); }
        else if (a == "--iter" && i+1<argc) { iterations = std::stoi(argv[++i]); }
        else if (a == "--threads" && i+1<argc) { threads = std::stoi(argv[++i]); }
        else if (a == "--rand") { rand_frames = true; }
        else if (a == "--gpu" && i+1<argc) { std::string v = argv[++i]; use_gpu = (v == "true" || v == "1"); }
        else if (a == "--mean" && i+3<argc) {
            mean[0] = std::stof(argv[++i]); mean[1] = std::stof(argv[++i]); mean[2] = std::stof(argv[++i]);
        }
        else if (a == "--std" && i+3<argc) {
            stdv[0] = std::stof(argv[++i]); stdv[1] = std::stof(argv[++i]); stdv[2] = std::stof(argv[++i]);
        }
        else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (src_w <= 0) src_w = in_w;
    if (src_h <= 0) src_h = in_h;

    std::cout << "Model: " << model_path << "\nMode: " << mode << "\nInput: " << in_w << "x" << in_h
              << "\nSource frames: " << src_w << "x" << src_h << " (precreated " << num_frames << " frames)\n"
              << "Warmup: " << warmup << ", Iterations: " << iterations << ", Threads: " << threads
              << ", GPU: " << (use_gpu ? "true" : "false") << "\n\n";

    if (threads > 0) {
        cv::setNumThreads(threads);
    };
    cv::setUseOptimized(true);

    std::cout << "[init] loading model...\n";
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 2;
    }

    if (use_gpu) {
        int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
        if (cudaCount > 0) {
            std::cout << "[init] CUDA devices found: " << cudaCount << ". Setting DNN backend to CUDA (FP16 attempt)...\n";
            try {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                std::cout << "[init] Preferable backend/target set to CUDA/FP16.\n";
            } catch (...) {
                std::cout << "[init] Warning: setting CUDA backend failed; falling back to CPU.\n";
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } else {
            std::cout << "[init] No CUDA-capable device found (cv::cuda::getCudaEnabledDeviceCount()==0). Using CPU.\n";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::cout << "[init] creating synthetic frames...\n";
    std::vector<cv::Mat> frames;
    frames.reserve(num_frames);
    for (int i = 0; i < num_frames; ++i) {
        cv::Mat f(src_h, src_w, CV_8UC3);
        if (rand_frames) cv::randu(f, 0, 255);
        else f = cv::Mat::zeros(src_h, src_w, CV_8UC3);
        frames.push_back(f);
    }

    std::vector<cv::Mat> precomputed_blobs;
    if (mode == "infer") {
        std::cout << "[init] precomputing input blobs for inference-only mode...\n";
        precomputed_blobs.reserve(num_frames);
        for (int i = 0; i < num_frames; ++i) {
            cv::Mat resized;
            if (frames[i].cols != in_w || frames[i].rows != in_h) cv::resize(frames[i], resized, cv::Size(in_w, in_h));
            else resized = frames[i];
            cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, cv::Size(in_w, in_h),
                                                  cv::Scalar(mean[0], mean[1], mean[2]), true, false);
            int channels = blob.size[1], height = blob.size[2], width = blob.size[3];
            for (int c = 0; c < channels; ++c) {
                cv::Mat plane(height, width, CV_32F, blob.ptr<float>(0, c));
                plane /= stdv[c];
            }
            precomputed_blobs.push_back(blob);
        }
    }

    cv::Mat output;
    std::cout << "[warmup] running " << warmup << " warmup iterations...\n";
    for (int w = 0; w < warmup; ++w) {
        int idx = w % num_frames;
        if (mode == "infer") {
            net.setInput(precomputed_blobs[idx]);
            output = net.forward();
        } else {
            cv::Mat resized;
            if (frames[idx].cols != in_w || frames[idx].rows != in_h) cv::resize(frames[idx], resized, cv::Size(in_w, in_h));
            else resized = frames[idx];
            cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, cv::Size(in_w, in_h),
                                                  cv::Scalar(mean[0], mean[1], mean[2]), true, false);
            for (int c = 0; c < static_cast<int>(stdv.size()); ++c) {
                cv::Mat plane(in_h, in_w, CV_32F, blob.ptr<float>(0, c));
                plane /= stdv[c];
            }
            net.setInput(blob);
            output = net.forward();
        }
    }

    std::cout << "[measure] starting measured loop (" << iterations << " iterations)...\n";
    std::vector<double> times_ms;
    times_ms.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
        int idx = it % num_frames;
        auto t0 = Clock::now();

        if (mode == "infer") {
            net.setInput(precomputed_blobs[idx]);
            output = net.forward();
        } else {
            cv::Mat resized;
            if (frames[idx].cols != in_w || frames[idx].rows != in_h) cv::resize(frames[idx], resized, cv::Size(in_w, in_h));
            else resized = frames[idx];
            cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, cv::Size(in_w, in_h),
                                                  cv::Scalar(mean[0], mean[1], mean[2]), true, false);
            for (int c = 0; c < static_cast<int>(stdv.size()); ++c) {
                cv::Mat plane(in_h, in_w, CV_32F, blob.ptr<float>(0, c));
                plane /= stdv[c];
            }
            net.setInput(blob);
            output = net.forward();
        }

        auto t1 = Clock::now();
        double ms = ms_d(t1 - t0).count();
        times_ms.push_back(ms);
    }

    double total_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean_ms = total_ms / times_ms.size();
    auto minmax = std::minmax_element(times_ms.begin(), times_ms.end());
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    double median_ms = sorted[sorted.size()/2];
    double p95_ms = sorted[static_cast<size_t>(sorted.size() * 0.95)];
    double fps = (iterations * 1000.0) / total_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- Benchmark results (" << mode << ") ---\n";
    std::cout << "Iterations: " << iterations << ", Frames-precreated: " << num_frames << "\n";
    std::cout << "Total time measured: " << total_ms << " ms\n";
    std::cout << "FPS (measured) : " << fps << " fps\n";
    std::cout << "mean latency   : " << mean_ms << " ms\n";
    std::cout << "median latency : " << median_ms << " ms\n";
    std::cout << "min latency    : " << *minmax.first << " ms\n";
    std::cout << "max latency    : " << *minmax.second << " ms\n";
    std::cout << "95%-tile       : " << p95_ms << " ms\n";

    std::cout << "\nFirst 10 measured latencies (ms):\n";
    for (size_t i = 0; i < std::min<size_t>(10, times_ms.size()); ++i)
        std::cout << times_ms[i] << " ";
    std::cout << "\n\nDone.\n";
    return 0;
}
