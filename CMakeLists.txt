cmake_minimum_required(VERSION 3.12)
project(ml_cpp)

# Add C++11 support
set(CMAKE_CXX_STANDARD 11)

# Add the source files for your library
set(SOURCES
    src/dataset.cpp
    src/kmeans.cpp
    src/linear_regression.cpp
    src/metrics.cpp
    src/minmax_scaler.cpp
    src/randomforest_classifier.cpp
    src/rf_node.cpp
    src/standard_scaler.cpp
    src/utils.cpp
    src/validation.cpp
)

# Add the header files for your library
set(HEADERS
    include/dataset.hpp
    include/kmeans.hpp
    include/linear_regression.hpp
    include/metrics.hpp
    include/minmax_scaler.hpp
    include/randomforest_classifier.hpp
    include/rf_node.hpp
    include/standard_scaler.hpp
    include/utils.hpp
    include/validation.hpp
)

# Add the executable for your tests
add_executable(tests 
            tests/test_dataset.cpp
            tests/test_kmeans.cpp
            tests/test_metrics.cpp 
            tests/test_minmax_scaler.cpp 
            tests/test_randomforest_classifier.cpp
            tests/test_rf_node.cpp
            tests/test_standard_scaler.cpp
            tests/test_utils.cpp)

add_executable(classification 
            examples/classification.cpp)

add_executable(regression 
            examples/regression.cpp)



find_package(NumCpp 2.12.1 REQUIRED)
find_package (Catch2 3 REQUIRED)

# Add the library target
add_library(ml_cpp ${SOURCES} ${HEADERS})

# Specify include directories for the library
target_include_directories(ml_cpp PUBLIC include)

# Link the library to the test executable
target_link_libraries(tests PRIVATE ml_cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)

# Link the library to the examples
target_link_libraries(classification PRIVATE ml_cpp)
target_link_libraries(classification PRIVATE NumCpp::NumCpp)

target_link_libraries(regression PRIVATE ml_cpp)
target_link_libraries(regression PRIVATE NumCpp::NumCpp)