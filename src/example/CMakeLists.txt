cmake_minimum_required(VERSION 3.10) 
project(vlsi-example) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Eigen3 REQUIRED)

set(CMAKE_PREFIX_PATH "/home/eebyao/libtorch/share/cmake/Torch")
set(PROJECT_ROOT_PATH "/home/eebyao/libtorch/share/cmake/Torch")

find_package(CUDAToolkit QUIET) 
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, enabling CUDA support")
    add_definitions(-DCUDA_ENABLE) 
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(WARNING "CUDA not found, disabling CUDA support")
endif()

# set(USE_CUDA OFF)          
# set(USE_CUDNN OFF)         
# set(USE_NCCL OFF)          

find_package(Torch REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cuda/Ising_bSB)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../tools/matplotlib-cpp)

include_directories(/home/byao/anaconda3/lib/python3.11/site-packages/numpy/core/include)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BStarTree.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BsbSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SatSolver.cc
)

add_executable(fp ${SOURCES})


target_link_libraries(fp 
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# 仅当 CUDA 可用时链接 CUDA 库
if(CUDAToolkit_FOUND)
    target_link_libraries(fp
        CUDA::cublas
        ${CUDA_cusparse_LIBRARY} 
        ${CUDA_LIBRARIES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/Ising_bSB/libbSB.so
    )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)