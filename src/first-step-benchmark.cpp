/**
 * High-Speed Trigger Algorithm Benchmark
 * Реализация алгоритма-триггера для двухэтапной системы детекции дефектов
 * 
 * Алгоритм Брэдли: Bradley, D., & Roth, G. (2007). Adaptive Thresholding 
 * Using the Integral Image. Journal of Graphics Tools.
 * 
 * Компиляция: g++ -O3 -std=c++17 -I/usr/include/opencv4 trigger-benchmark.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -o trigger-benchmark
 */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <iomanip>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using ms_d = std::chrono::duration<double, std::milli>;

// Конфигурация алгоритма
struct TriggerConfig {
    int window_size = 41;          // Размер окна для алгоритма Брэдли
    float sensitivity = 0.15f;     // Параметр чувствительности t
    double min_area = 25.0;        // Минимальная площадь области
    double max_area = 10000.0;     // Максимальная площадь области  
    double min_compactness = 1.2;  // Минимальная компактность
    double max_compactness = 10.0; // Максимальная компактность
    double min_eccentricity = 0.3; // Минимальный эксцентриситет
    double max_eccentricity = 0.95;// Максимальный эксцентриситет
};

/**
 * Реализация алгоритма Брэдли для адаптивной бинаризации
 * Источник: Bradley, D., & Roth, G. (2007)
 */
void bradley_threshold(const cv::Mat& src, cv::Mat& dst, int window_size, float t) {
    CV_Assert(src.type() == CV_8UC1);
    
    //  интегральное изображение
    cv::Mat integral;
    cv::integral(src, integral, CV_32S);
    
    dst.create(src.size(), CV_8UC1);
    int half_window = window_size / 2;
    
    for (int r = 0; r < src.rows; ++r) {
        int top = std::max(0, r - half_window);
        int bottom = std::min(src.rows - 1, r + half_window);
        
        for (int c = 0; c < src.cols; ++c) {
            int left = std::max(0, c - half_window);
            int right = std::min(src.cols - 1, c + half_window);
            
            // сумма в окне за O(1) с использованием интегрального изображения
            int sum = integral.at<int>(bottom + 1, right + 1) 
                    - integral.at<int>(top, right + 1) 
                    - integral.at<int>(bottom + 1, left) 
                    + integral.at<int>(top, left);
            
            int area = (bottom - top + 1) * (right - left + 1);
            int threshold = static_cast<int>(sum * t);
            
            dst.at<uchar>(r, c) = (src.at<uchar>(r, c) < threshold) ? 0 : 255;
        }
    }
}

/**
 * Вычисление компактности (отношение квадрата периметра к площади)
 * Стандартная метрика формы в компьютерном зрении
 */
double compute_compactness(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (area <= std::numeric_limits<double>::epsilon()) return 0.0;
    return (perimeter * perimeter) / (4 * CV_PI * area);
}

/**
 * Вычисление эксцентриситета через аппроксимацию эллипсом
 * Стандартная метрика в OpenCV
 */
double compute_eccentricity(const std::vector<cv::Point>& contour) {
    if (contour.size() < 5) return 0.0;
    
    try {
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        double a = std::max(ellipse.size.width, ellipse.size.height) / 2.0;
        double b = std::min(ellipse.size.width, ellipse.size.height) / 2.0;
        
        if (a < std::numeric_limits<double>::epsilon()) return 0.0;
        return std::sqrt(1.0 - (b * b) / (a * a));
    } catch (...) {
        return 0.0;
    }
}

/**
 * Основной конвейер обработки алгоритма-триггера
 */
int process_trigger_pipeline(const cv::Mat& frame, const TriggerConfig& config) {
    cv::Mat gray, binary;
    
    // 1. Конвертация в grayscale (REC.601 стандарт)
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // 2. Компенсация освещения через Gaussian blur
    cv::Mat background;
    cv::GaussianBlur(gray, background, cv::Size(51, 51), 0);
    cv::Mat normalized;
    cv::divide(gray, background, normalized, 1.0, CV_8UC1);
    
    // 3. Адаптивная бинаризация по Брэдли
    bradley_threshold(normalized, binary, config.window_size, config.sensitivity);
    
    // 4. Морфологическое закрытие для объединения областей
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    
    // 5. Поиск контуров
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 6. Каскадный анализ с быстрым отказом
    int defect_candidates = 0;
    
    for (const auto& contour : contours) {
        // Первая проверка: фильтрация по размеру
        double area = cv::contourArea(contour);
        if (area < config.min_area || area > config.max_area) continue;
        
        // Вторая проверка: эксцентриситет (быстрая проверка)
        double ecc = compute_eccentricity(contour);
        if (ecc < config.min_eccentricity || ecc > config.max_eccentricity) continue;
        
        // Третья проверка: компактность (более сложная проверка)
        double compact = compute_compactness(contour);
        if (compact < config.min_compactness || compact > config.max_compactness) continue;
        
        defect_candidates++;
    }
    
    return defect_candidates;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --width N              Frame width (default: 640)\n"
              << "  --height N             Frame height (default: 480)\n" 
              << "  --frames N             Number of test frames (default: 100)\n"
              << "  --warmup N             Warmup iterations (default: 10)\n"
              << "  --iter N               Measurement iterations (default: 1000)\n"
              << "  --window N             Bradley window size (default: 41)\n"
              << "  --sensitivity F        Bradley sensitivity (default: 0.15)\n"
              << "  --defect-ratio F       Ratio of frames with synthetic defects (default: 0.3)\n";
}

int main(int argc, char** argv) {
    // Параметры по умолчанию
    int width = 640;
    int height = 480; 
    int num_frames = 100;
    int warmup_iter = 10;
    int measure_iter = 1000;
    float defect_ratio = 0.3f;
    TriggerConfig config;
    
    // Парсинг аргументов
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) num_frames = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup_iter = std::stoi(argv[++i]);
        else if (arg == "--iter" && i + 1 < argc) measure_iter = std::stoi(argv[++i]);
        else if (arg == "--window" && i + 1 < argc) config.window_size = std::stoi(argv[++i]);
        else if (arg == "--sensitivity" && i + 1 < argc) config.sensitivity = std::stof(argv[++i]);
        else if (arg == "--defect-ratio" && i + 1 < argc) defect_ratio = std::stof(argv[++i]);
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
    }
    
    std::cout << "=== High-Speed Trigger Algorithm Benchmark ===\n"
              << "Resolution: " << width << "x" << height << "\n"
              << "Test frames: " << num_frames << " (defect ratio: " << defect_ratio * 100 << "%)\n"
              << "Warmup: " << warmup_iter << ", Measurements: " << measure_iter << "\n"
              << "Bradley params: window=" << config.window_size << ", sensitivity=" << config.sensitivity << "\n\n";
    
    // Инициализация OpenCV
    cv::setUseOptimized(true);
    cv::setNumThreads(0); // Автоматическое определение потоков
    
    // Генерация тестовых данных
    std::cout << "[init] Generating synthetic test frames...\n";
    std::vector<cv::Mat> test_frames;
    test_frames.reserve(num_frames);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> defect_gen(0.0f, 1.0f);
    std::uniform_int_distribution<int> pos_gen(50, width - 50);
    
    for (int i = 0; i < num_frames; ++i) {
        cv::Mat frame(height, width, CV_8UC3);
        
        // Базовый текстурный фон (имитация металлической поверхности)
        cv::randu(frame, cv::Scalar(100, 100, 100), cv::Scalar(200, 200, 200));
        
        // Добавление синтетических дефектов на часть кадров
        if (defect_gen(gen) < defect_ratio) {
            int defect_type = i % 3;
            
            switch (defect_type) {
                case 0: // Точечный дефект (включение)
                    cv::circle(frame, cv::Point(pos_gen(gen), pos_gen(gen)), 
                             8, cv::Scalar(50, 50, 50), -1);
                    break;
                case 1: // Линейный дефект (царапина)
                    cv::line(frame, 
                             cv::Point(pos_gen(gen), pos_gen(gen)),
                             cv::Point(pos_gen(gen), pos_gen(gen)),
                             cv::Scalar(30, 30, 30), 3);
                    break;
                case 2: // Пятно
                    cv::ellipse(frame, 
                               cv::Point(pos_gen(gen), pos_gen(gen)),
                               cv::Size(15, 8), 0, 0, 360, 
                               cv::Scalar(70, 70, 70), -1);
                    break;
            }
        }
        
        test_frames.push_back(frame.clone());
    }
    
    // Прогревочные итерации
    std::cout << "[warmup] Running " << warmup_iter << " warmup iterations...\n";
    for (int i = 0; i < warmup_iter; ++i) {
        int idx = i % num_frames;
        process_trigger_pipeline(test_frames[idx], config);
    }
    
    // Основные измерения
    std::cout << "[measure] Starting measurement loop (" << measure_iter << " iterations)...\n";
    std::vector<double> latencies;
    std::vector<int> candidates_detected;
    latencies.reserve(measure_iter);
    candidates_detected.reserve(measure_iter);
    
    for (int i = 0; i < measure_iter; ++i) {
        int idx = i % num_frames;
        
        auto start_time = Clock::now();
        int candidates = process_trigger_pipeline(test_frames[idx], config);
        auto end_time = Clock::now();
        
        double latency = ms_d(end_time - start_time).count();
        latencies.push_back(latency);
        candidates_detected.push_back(candidates);
    }
    
    // Статистика
    double total_time = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean_latency = total_time / latencies.size();
    double fps = 1000.0 / mean_latency;
    
    std::nth_element(latencies.begin(), latencies.begin() + latencies.size() / 2, latencies.end());
    double median_latency = latencies[latencies.size() / 2];
    
    std::vector<double> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    double p95_latency = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.95)];
    
    int total_candidates = std::accumulate(candidates_detected.begin(), candidates_detected.end(), 0);
    
    // Вывод результатов
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Benchmark Results ===\n"
              << "Total processing time: " << total_time << " ms\n"
              << "Frames processed: " << measure_iter << "\n"
              << "Mean latency: " << mean_latency << " ms\n"
              << "Median latency: " << median_latency << " ms\n"
              << "95th percentile: " << p95_latency << " ms\n"
              << "Throughput: " << fps << " FPS\n"
              << "Total candidates detected: " << total_candidates << "\n"
              << "Avg candidates per frame: " << static_cast<double>(total_candidates) / measure_iter << "\n";
    
    // Проверка достижения целевых показателей
    std::cout << "\n=== Target Compliance ===\n";
    std::cout << "Target FPS: 200.0\n";
    std::cout << "Achieved FPS: " << fps << "\n";
    std::cout << "Status: " << (fps >= 200.0 ? "PASSED" : "FAILED") << "\n";
    
    return 0;
}