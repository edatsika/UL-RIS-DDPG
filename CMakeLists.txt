cmake_minimum_required(VERSION 3.10)
project(DDPG_Libtorch)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Specify the path to Libtorch
set(Torch_DIR "/home/user/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Add executable
add_executable(DDPG main.cpp DDPG.cpp env.cpp)

# Link Libtorch libraries
target_link_libraries(DDPG "${TORCH_LIBRARIES}")

# Include headers (including API headers)
target_include_directories(DDPG PRIVATE
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_INCLUDE_DIRS}/../torch/csrc/api/include"
)

# Ensure that the RPATH is set correctly to find Libtorch at runtime
set_property(TARGET DDPG PROPERTY CXX_STANDARD 17)
set_property(TARGET DDPG PROPERTY BUILD_RPATH "/home/user/libtorch/lib")
