/**
 * High-Speed Trigger Algorithm Benchmark (First Stage Only)
 * Тестирование производительности первого этапа алгоритма-триггера
 * 
 * Алгоритм Брэдли: Bradley, D., & Roth, G. (2007). Adaptive Thresholding 
 * Using the Integral Image. Journal of Graphics Tools.
 * 
 * Компиляция: g++ -O3 -std=c++17 -I/usr/include/opencv4 first-step-benchmark.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -o first-step-benchmark
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
#include <filesystem>

using Clock = std::chrono::high_resolution_clock;
using ms_d = std::chrono::duration<double, std::milli>;
namespace fs = std::filesystem;

// Конфигурация алгоритма-триггера (только первый этап)
struct TriggerConfig {
    int bradley_window = 41;          // Размер окна для алгоритма Брэдли
    float bradley_sensitivity = 0.15f; // Чувствительность алгоритма Брэдли
};

/**
 * ПЕРВЫЙ ЭТАП: Алгоритм Брэдли для адаптивной бинаризации
 * С добавлением отладки
 */
void bradley_threshold(const cv::Mat& src, cv::Mat& dst, int window_size, float t) {
    CV_Assert(src.type() == CV_8UC1);
    
    // Вычисление интегрального изображения
    cv::Mat integral;
    cv::integral(src, integral, CV_32S);
    
    dst.create(src.size(), CV_8UC1);
    int half_window = window_size / 2;
    
    int total_pixels_above = 0;
    int total_pixels_below = 0;
    
    for (int r = 0; r < src.rows; ++r) {
        int top = std::max(0, r - half_window);
        int bottom = std::min(src.rows - 1, r + half_window);
        
        for (int c = 0; c < src.cols; ++c) {
            int left = std::max(0, c - half_window);
            int right = std::min(src.cols - 1, c + half_window);
            
            int sum = integral.at<int>(bottom + 1, right + 1) 
                    - integral.at<int>(top, right + 1) 
                    - integral.at<int>(bottom + 1, left) 
                    + integral.at<int>(top, left);
            
            int area = (bottom - top + 1) * (right - left + 1);
            
            // ИСПРАВЛЕНИЕ: Правильное вычисление порога
            // Среднее значение в окне = sum / area
            // Порог = среднее * чувствительность
            int mean_value = sum / area;
            int threshold = static_cast<int>(mean_value * t);
            
            // ИСПРАВЛЕНИЕ: Инвертируем логику для царапин
            // Царапины обычно ТЕМНЕЕ фона, поэтому должны быть ЧЕРНЫМИ
            if (src.at<uchar>(r, c) < threshold) {
                dst.at<uchar>(r, c) = 0;    // Черный - потенциальный дефект
                total_pixels_below++;
            } else {
                dst.at<uchar>(r, c) = 255;  // Белый - фон
                total_pixels_above++;
            }
        }
    }
    
    // Отладочная информация
    std::cout << "Bradley threshold stats: " << std::endl;
    std::cout << "  Pixels below threshold (potential defects): " << total_pixels_below << std::endl;
    std::cout << "  Pixels above threshold (background): " << total_pixels_above << std::endl;
    std::cout << "  Defect ratio: " << (double)total_pixels_below / (total_pixels_above + total_pixels_below) * 100.0 << "%" << std::endl;
}
/**
 * ПЕРВЫЙ ЭТАП: Основной конвейер обработки
 * Включает только предобработку и бинаризацию
 */
bool process_first_stage(const cv::Mat& frame, const TriggerConfig& config, bool demo_mode = false) {
    if (demo_mode) {
        cv::Mat display_frame = frame.clone();
        cv::putText(display_frame, "1. Original Image", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Processing Steps", display_frame);
        cv::waitKey(0);
    }
    
    // Конвертация в grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Легкое размытие для уменьшения шума
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    
    if (demo_mode) {
        cv::Mat display_blurred;
        cv::cvtColor(blurred, display_blurred, cv::COLOR_GRAY2BGR);
        cv::putText(display_blurred, "2. After Gaussian Blur (3x3)", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Processing Steps", display_blurred);
        cv::waitKey(0);
    }
    
    // Бинаризация по исправленному алгоритму Брэдли
    cv::Mat binary_bradley;
    cv::adaptiveThreshold(blurred, binary_bradley, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    // bradley_threshold(blurred, binary_bradley, config.bradley_window, config.bradley_sensitivity);
    
    if (demo_mode) {
        cv::Mat display_bradley;
        cv::cvtColor(binary_bradley, display_bradley, cv::COLOR_GRAY2BGR);
        cv::putText(display_bradley, "3. Bradley Binary", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Processing Steps", display_bradley);
        cv::waitKey(0);
    }
    
    // // АГРЕССИВНАЯ ФИЛЬТРАЦИЯ ШУМА
    // cv::Mat cleaned;
    
    // // 1. Морфологическое закрытие для объединения близких областей
    // cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    // cv::morphologyEx(binary_bradley, cleaned, cv::MORPH_CLOSE, kernel_close);
    
    // // 2. Морфологическое открытие для удаления мелких объектов
    // cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    // cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel_open);
    
    // if (demo_mode) {
    //     cv::Mat display_cleaned;
    //     cv::cvtColor(cleaned, display_cleaned, cv::COLOR_GRAY2BGR);
    //     cv::putText(display_cleaned, "4. After Noise Filtering", cv::Point(10, 30), 
    //                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    //     cv::imshow("Processing Steps", display_cleaned);
    //     cv::waitKey(0);
    // }
    
    // Поиск контуров с фильтрацией по размеру
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_bradley, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // ФИЛЬТРАЦИЯ КОНТУРОВ ПО РАЗМЕРУ И ФОРМЕ
    std::vector<std::vector<cv::Point>> filtered_contours;
    int min_contour_area = 50;  // Минимальная площадь контура
    int max_contour_area = 5000; // Максимальная площадь контура
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= min_contour_area && area <= max_contour_area) {
            // Дополнительная фильтрация по форме (опционально)
            cv::Rect bbox = cv::boundingRect(contour);
            double aspect_ratio = (double)bbox.width / bbox.height;
            
            // Фильтруем слишком вытянутые или слишком квадратные объекты
            if (aspect_ratio >= 0.2 && aspect_ratio <= 5.0) {
                filtered_contours.push_back(contour);
            }
        }
    }
    
    if (demo_mode) {
        std::cout << "Contour filtering: " << contours.size() << " -> " << filtered_contours.size() << " contours" << std::endl;
    }
    
    // Визуализация результатов
    if (demo_mode) {
        cv::Mat result = frame.clone();
        
        if (!filtered_contours.empty()) {
            cv::drawContours(result, filtered_contours, -1, cv::Scalar(0, 255, 0), 2);
            
            std::string info = "Filtered contours: " + std::to_string(filtered_contours.size());
            cv::putText(result, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(result, "TRIGGER ACTIVATED", cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(result, "No significant contours - NO TRIGGER", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        cv::imshow("Processing Steps", result);
        std::cout << "Press any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    return !filtered_contours.empty();
}
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --width N              Frame width (default: 640)\n"
              << "  --height N             Frame height (default: 480)\n" 
              << "  --frames N             Number of test frames (default: 50)\n"
              << "  --warmup N             Warmup iterations (default: 10)\n"
              << "  --iter N               Measurement iterations (default: 500)\n"
              << "  --window N             Bradley window size (default: 41)\n"
              << "  --sensitivity F        Bradley sensitivity (default: 0.15)\n"
              << "  --defect-ratio F       Ratio of frames with synthetic defects (default: 0.3)\n"
              << "  --demo                 Enable demonstration mode (shows processing steps)\n"
              << "  --input image.jpg      Process specific JPEG image file\n";
}

int main(int argc, char** argv) {
    // Параметры по умолчанию (аналогично предыдущему тесту)
    int width = 640;
    int height = 480; 
    int num_frames = 50;
    int warmup_iter = 10;
    int measure_iter = 500;
    float defect_ratio = 0.3f;
    bool demo_mode = false;
    std::string input_image_path;
    TriggerConfig config;
    
    // Парсинг аргументов
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) num_frames = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup_iter = std::stoi(argv[++i]);
        else if (arg == "--iter" && i + 1 < argc) measure_iter = std::stoi(argv[++i]);
        else if (arg == "--window" && i + 1 < argc) config.bradley_window = std::stoi(argv[++i]);
        else if (arg == "--sensitivity" && i + 1 < argc) config.bradley_sensitivity = std::stof(argv[++i]);
        else if (arg == "--defect-ratio" && i + 1 < argc) defect_ratio = std::stof(argv[++i]);
        else if (arg == "--demo") demo_mode = true;
        else if (arg == "--input" && i + 1 < argc) input_image_path = argv[++i];
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
    }
    
    // Режим обработки конкретного изображения
    if (!input_image_path.empty()) {
        // Проверяем существование файла
        if (!fs::exists(input_image_path)) {
            std::cerr << "Error: Input file '" << input_image_path << "' not found!\n";
            return 1;
        }
        
        // Проверяем расширение файла
        std::string extension = fs::path(input_image_path).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension != ".jpg" && extension != ".jpeg") {
            std::cerr << "Error: Only JPEG images are supported. File has extension: " << extension << "\n";
            return 1;
        }
        
        // Загружаем изображение
        cv::Mat image = cv::imread(input_image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image from '" << input_image_path << "'\n";
            return 1;
        }
        
        // Выводим информацию о исходном изображении
        std::cout << "=== Processing Input Image ===\n";
        std::cout << "Input file: " << input_image_path << "\n";
        std::cout << "Original size: " << image.cols << "x" << image.rows << "\n";
        
        // Изменяем размер если необходимо
        if (image.cols != 640 || image.rows != 480) {
            cv::resize(image, image, cv::Size(640, 480));
            std::cout << "Resized to: 640x480\n";
        } else {
            std::cout << "No resizing needed\n";
        }
        
        std::cout << "Bradley params: window=" << config.bradley_window 
                  << ", sensitivity=" << config.bradley_sensitivity << "\n\n";
        
        // Обрабатываем изображение в демо-режиме
        bool trigger_activated = process_first_stage(image, config, true);
        std::cout << "Trigger activated: " << (trigger_activated ? "YES" : "NO") << "\n";
        
        cv::destroyAllWindows();
        return 0;
    }
    
    // В демо-режиме меняем параметры для наглядности
    if (demo_mode) {
        num_frames = 1;  // Только один кадр для демонстрации
        measure_iter = 1;
        warmup_iter = 0;
        defect_ratio = 1.0f; // Всегда с дефектом
        std::cout << "=== DEMO MODE: First Stage Trigger Algorithm ===\n";
    } else {
        std::cout << "=== First Stage Trigger Algorithm Benchmark ===\n";
    }
    
    std::cout << "Resolution: " << width << "x" << height << "\n"
              << "Test frames: " << num_frames << " (defect ratio: " << defect_ratio * 100 << "%)\n"
              << "Warmup: " << warmup_iter << ", Measurements: " << measure_iter << "\n"
              << "Bradley params: window=" << config.bradley_window << ", sensitivity=" << config.bradley_sensitivity << "\n\n";
    
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
    std::uniform_int_distribution<int> pos_x(50, width - 50);
    std::uniform_int_distribution<int> pos_y(50, height - 50);
    
    for (int i = 0; i < num_frames; ++i) {
        cv::Mat frame(height, width, CV_8UC3);
        
        // Базовый текстурный фон (имитация металлической поверхности)
        cv::randu(frame, cv::Scalar(100, 100, 100), cv::Scalar(200, 200, 200));
        
        // Добавление синтетических дефектов на часть кадров
        if (defect_gen(gen) < defect_ratio) {
            int defect_type = i % 3;
            
            switch (defect_type) {
                case 0: // Точечный дефект (включение)
                    cv::circle(frame, cv::Point(pos_x(gen), pos_y(gen)), 
                             8, cv::Scalar(50, 50, 50), -1);
                    break;
                case 1: // Линейный дефект (царапина)
                    cv::line(frame, 
                             cv::Point(pos_x(gen), pos_y(gen)),
                             cv::Point(pos_x(gen) + 30, pos_y(gen)),
                             cv::Scalar(30, 30, 30), 2);
                    break;
                case 2: // Пятно
                    cv::ellipse(frame, 
                               cv::Point(pos_x(gen), pos_y(gen)),
                               cv::Size(15, 8), 0, 0, 360, 
                               cv::Scalar(70, 70, 70), -1);
                    break;
            }
        }
        
        test_frames.push_back(frame.clone());
    }
    
    // В демо-режиме показываем процесс, в обычном - делаем замеры
    if (demo_mode) {
        std::cout << "[demo] Showing processing steps for first frame...\n";
        std::cout << "Press any key to advance to next processing stage...\n";
        bool trigger_activated = process_first_stage(test_frames[0], config, true);
        std::cout << "Trigger activated: " << (trigger_activated ? "YES" : "NO") << "\n";
        cv::destroyAllWindows();
        return 0;
    }
    
    // Прогревочные итерации
    std::cout << "[warmup] Running " << warmup_iter << " warmup iterations...\n";
    for (int i = 0; i < warmup_iter; ++i) {
        int idx = i % num_frames;
        process_first_stage(test_frames[idx], config);
    }
    
    // Основные измерения
    std::cout << "[measure] Starting measurement loop (" << measure_iter << " iterations)...\n";
    std::vector<double> latencies;
    std::vector<bool> triggers_activated;
    latencies.reserve(measure_iter);
    triggers_activated.reserve(measure_iter);
    
    for (int i = 0; i < measure_iter; ++i) {
        int idx = i % num_frames;
        
        auto start_time = Clock::now();
        bool trigger_activated = process_first_stage(test_frames[idx], config);
        auto end_time = Clock::now();
        
        double latency = ms_d(end_time - start_time).count();
        latencies.push_back(latency);
        triggers_activated.push_back(trigger_activated);
    }
    
    // Статистика
    double total_time = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean_latency = total_time / latencies.size();
    double fps = 1000.0 / mean_latency;
    
    std::vector<double> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    double median_latency = sorted_latencies[sorted_latencies.size() / 2];
    double p95_latency = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.95)];
    
    int total_triggers = std::count(triggers_activated.begin(), triggers_activated.end(), true);
    double trigger_rate = (static_cast<double>(total_triggers) / measure_iter) * 100.0;
    
    // Вывод результатов
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== Benchmark Results ===\n"
              << "Total processing time: " << total_time << " ms\n"
              << "Frames processed: " << measure_iter << "\n"
              << "Mean latency: " << mean_latency << " ms\n"
              << "Median latency: " << median_latency << " ms\n"
              << "95th percentile: " << p95_latency << " ms\n"
              << "Throughput: " << fps << " FPS\n"
              << "Triggers activated: " << total_triggers << " (" << trigger_rate << "% of frames)\n";
    
    // Проверка достижения целевых показателей
    std::cout << "\n=== Target Compliance ===\n";
    std::cout << "Target FPS: 200.0\n";
    std::cout << "Achieved FPS: " << fps << "\n";
    std::cout << "Status: " << (fps >= 200.0 ? "PASSED" : "FAILED") << "\n";
    
    return 0;
}