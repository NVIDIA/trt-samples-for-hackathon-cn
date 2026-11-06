/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// mmio_probe —— 配合宿主机侧读取 KVM 的 mmio_exits 计数器使用。
//
// 目的：验证「kernel launch 末端的 doorbell MMIO 写是否被 KVM trap」。
// 每次 launch 理论上对应一次 doorbell 写；若被 trap，宿主机的 mmio_exits
// 在本程序运行前后应增加约 N。
//
// 本程序只做一件事：**精确发射 N 次空 kernel**，前后各留一段静默窗口，
// 便于宿主机侧划定采样区间。不做任何统计、不做多余的 CUDA 调用。
//
// 用法：  ./mmio_probe [N] [--pause 秒]
//   默认 N=100000，前后各静默 3 秒。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define CK(x)                                                                                    \
    do                                                                                           \
    {                                                                                            \
        cudaError_t e = (x);                                                                     \
        if (e != cudaSuccess)                                                                    \
        {                                                                                        \
            std::fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

__global__ void empty_kernel() {}

static void nap(int sec)
{
    struct timespec ts
    {
        sec, 0
    };
    nanosleep(&ts, nullptr);
}
static void stamp(const char *what)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    char b[32];
    strftime(b, sizeof b, "%H:%M:%S", &tm);
    std::printf("[%s.%03ld] %s\n", b, ts.tv_nsec / 1000000, what);
    std::fflush(stdout);
}

int main(int argc, char **argv)
{
    long N       = 100000;
    int  pause_s = 3;
    int  device  = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!std::strcmp(argv[i], "--pause") && i + 1 < argc)
            pause_s = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
            device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--help"))
        {
            std::printf("用法: %s [N] [--pause 秒] [--device N]\n", argv[0]);
            return 0;
        }
        else
            N = std::atol(argv[i]);
    }
    if (N < 1)
    {
        std::fprintf(stderr, "N 必须 >= 1\n");
        return 1;
    }

    CK(cudaSetDevice(device));
    cudaDeviceProp p {};
    CK(cudaGetDeviceProperties(&p, device));
    std::printf("GPU: %s | device=%d\n", p.name, device);
    std::printf("本次将精确发射 %ld 次空 kernel\n\n", N);

    // 先把 CUDA 上下文、模块加载等一次性开销做完，避免计入采样区间
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    for (int i = 0; i < 2000; i++)
        empty_kernel<<<1, 1, 0, s>>>();
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s));

    stamp("=== 静默窗口开始（宿主机请在此期间读取计数器）===");
    nap(pause_s);

    stamp("=== 发射开始 ===");
    for (long i = 0; i < N; i++)
        empty_kernel<<<1, 1, 0, s>>>();
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(s));
    stamp("=== 发射结束 ===");

    nap(pause_s);
    stamp("=== 静默窗口结束（宿主机请在此期间再次读取计数器）===");

    std::printf("\n实际发射次数: %ld\n", N);
    std::printf("若 doorbell 写被 KVM trap，宿主机 mmio_exits 的增量应约为 %ld\n", N);
    CK(cudaStreamDestroy(s));
    return 0;
}
