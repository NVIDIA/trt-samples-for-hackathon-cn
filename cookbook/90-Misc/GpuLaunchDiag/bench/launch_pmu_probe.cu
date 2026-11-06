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

// 精确发射 T 次空 kernel，用于配合 freq_monitor 做「两点差分」求每次 launch 的
// 指令数 / 周期数 / 耗时。
//
// 为什么要分批同步：背靠背发射到一定量后发射队列填满，CPU 会被背压阻塞并自旋等待，
// 这会把指令数也一起带偏。按 B 次一批同步可以把队列深度压在安全范围内。
// 只要两次运行用同一个 B，这部分开销按每次 launch 摊薄后是常量，差分时自动消掉。
//
// 用法: launch_pmu_probe <总发射次数 T> [--batch B=1024] [--device N] [--warmup W=20000]
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char **argv)
{
    long T = 0, B = 1024, W = 20000;
    int  device = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!std::strcmp(argv[i], "--batch") && i + 1 < argc)
            B = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
            device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc)
            W = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--help"))
        {
            std::printf("用法: %s <总发射次数 T> [--batch B=1024] [--device N] [--warmup W=20000]\n", argv[0]);
            return 0;
        }
        else
            T = std::atol(argv[i]);
    }
    if (T < 1 || B < 1)
    {
        std::fprintf(stderr, "T 和 B 必须 >= 1\n");
        return 1;
    }

    CK(cudaSetDevice(device));
    // ★ 自报实际加载的 UMD ★
    // nvcc 默认静态链接 cudart，libcuda 是运行时 dlopen 进来的，
    // 所以外部用 ldd 看不到。必须由进程自己报告，否则无法确认
    // 跑的到底是宿主机的 UMD 还是容器里 cuda-compat 提供的那个。
    {
        int dv = 0;
        cudaDriverGetVersion(&dv);
        std::printf("driver_API_version=%d  (= 1000*major + 10*minor)\n", dv);
        FILE *m = std::fopen("/proc/self/maps", "r");
        if (m)
        {
            char line[4096];
            char seen[4096] = "";
            while (std::fgets(line, sizeof line, m))
            {
                char *q = std::strstr(line, "libcuda.so");
                if (!q)
                    continue;
                char *nl = std::strchr(line, '\n');
                if (nl)
                    *nl = 0;
                char       *sp   = std::strrchr(line, ' ');
                const char *path = sp ? sp + 1 : line;
                if (std::strcmp(seen, path))
                {
                    std::printf("已加载 libcuda: %s\n", path);
                    std::snprintf(seen, sizeof seen, "%s", path);
                }
            }
            std::fclose(m);
        }
    }
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // 预热：两次运行用同一个 W，差分时消掉
    for (long i = 0; i < W; i++)
        empty_kernel<<<1, 1, 0, s>>>();
    CK(cudaStreamSynchronize(s));

    long done = 0;
    while (done < T)
    {
        long n = (T - done < B) ? (T - done) : B;
        for (long i = 0; i < n; i++)
            empty_kernel<<<1, 1, 0, s>>>();
        CK(cudaStreamSynchronize(s));
        done += n;
    }
    CK(cudaGetLastError());
    std::printf("GPU=%s device=%d 发射=%ld batch=%ld warmup=%ld\n", p.name, device, T, B, W);
    return 0;
}
