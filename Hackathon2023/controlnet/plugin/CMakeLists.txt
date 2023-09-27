cmake_minimum_required(VERSION 3.13)
project(plugin C CXX)

set(TRT_ROOT_DIR "/home/player/TensorRT-8.6.1.6")

add_definitions(-DENABLE_SM75)
add_definitions(-DENABLE_SM80)
add_definitions(-DENABLE_SM86)
add_definitions(-DENABLE_SM89)

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    if (CMAKE_BUILD_TYPE MATCHES "Debug" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -w -Wl,-Bsymbolic")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -w -Wl,-Bsymbolic")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
    endif()
endif()

find_package(CUDA 10.2 QUIET)
message(STATUS "CUDA detected: " ${CUDA_VERSION})

set(CUDA_NVCC_FLAGS
        --disable-warnings;
        -O3;
        -gencode arch=compute_75,code=sm_75;  # RTX2060/2070/2080/2080Ti, Tesla T4, GTX1660
        -gencode arch=compute_80,code=sm_80;  # RTX2060/2070/2080/2080Ti, Tesla T4, GTX1660
        -gencode arch=compute_86,code=sm_86;  # RTX3060/3070/3080/3080Ti/3090, A1000/2000/3000/4000
        -gencode arch=compute_89,code=sm_89;  # RTX3060/3070/3080/3080Ti/3090, A1000/2000/3000/4000
)


set(SRC_FILE_EXTENSIONS h hpp cuh c cpp cu cc)
set(LINT_DIRS
        common
        groupNormPlugin
)

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        # set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
        foreach(f_var ${FOUND_FILES})
            set(exist FALSE)
            foreach(exclude_dir ${EXCLUDE_DIRS})
                string(FIND ${f_var} "/${exclude_dir}/" output_var REVERSE)
                if(NOT ${output_var} MATCHES -1)
                    set(exist TRUE)
                endif()
            endforeach()

            if(NOT exist)
                list(APPEND PLUGIN_SOURCES ${f_var})
            endif()
        endforeach()
    endforeach()
endforeach()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TRT_ROOT_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(${TRT_ROOT_DIR}/lib)

list(APPEND LINKER_LIBS nvinfer)
list(APPEND LINKER_LIBS cudnn cublas cudart)

cuda_add_library(plugin SHARED ${PLUGIN_SOURCES})
target_link_libraries(plugin ${LINKER_LIBS})
