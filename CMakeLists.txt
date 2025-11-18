cmake_minimum_required(VERSION 3.31)
project(resnet-benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

#add_executable(prod_recog src/benchmark.cpp)

add_executable(first_step_bench src/first-step-benchmark.cpp)

#target_include_directories(prod_recog PRIVATE ${OpenCV_INCLUDE_DIRS} )
#target_link_libraries(prod_recog PRIVATE ${OpenCV_LIBS} ${Tesseract_LIBRARIES})

target_include_directories(first_step_bench PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(first_step_bench PRIVATE ${OpenCV_LIBS} ${Tesseract_LIBRARIES})
