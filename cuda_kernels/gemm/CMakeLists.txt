cmake_minimum_required(VERSION 3.21)
# RTX 3060 Laptop: compute capability 8.6. 可在配置时用 -D 覆盖。
set(CMAKE_CUDA_ARCHITECTURES 86-real CACHE STRING "CUDA GPU architectures")
project(cuda_learning LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

foreach(example IN ITEMS gemm gemm_tiled gemm_wmma rmsnorm)
    add_executable(${example} src/${example}.cu)
    target_compile_options(${example} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>)
endforeach()

target_link_libraries(gemm_tiled PRIVATE CUDA::cublas)

add_executable(gemm_bench src/gemm_bench.cu)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)
target_compile_options(gemm_bench PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>)
