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

// launch_roundtrip —— 单次 kernel 发射往返的「GPU 时间轴 / 主机侧」拆分
//
// 由 kernel_launch_bench_split.cu 的测项 [5] 改进而来：原版只量 CUDA Event 之间的
// GPU 时间轴区间，看不到命令提交延迟和完成通知延迟。本程序在同一轮外面再套一层
// steady_clock，把这两部分分开：
//
//     wall  = 整轮墙钟（提交 3 条命令 + 等完成通知）
//     event = GPU 时间轴上 begin event -> end event 的区间
//     host  = wall - event   ← 不在 GPU 时间轴上的主机侧开销
//
// 注意：这是「一次一停」的模式，每轮都会把流水线排空，衡量的是单次往返延迟，
// 不是稳态吞吐，不能与背靠背发射的摊薄成本直接比较。
//
// 第二部分：纯 CPU 发射成本（按批量 N 扫描）
//
//     cudaStreamSynchronize(s);            // 先清空此前的工作
//     t0; for (N 次) empty_kernel<<<>>>(); t1;   // 只发射，不等
//     cudaStreamSynchronize(s);  t2;       // 排空
//
//     issue = (t1-t0)/N   每次发射的 CPU 成本
//     drain = (t2-t1)     发射完之后还要等多久排空
//
// 关键：N 一大，发射队列填满后 CPU 会被背压阻塞，issue 就不再是纯 CPU 成本，
// 而是退化成 GPU 的排空速率。所以必须扫描 N，并观察拐点在哪里。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                                                 \
    do                                                                                                                   \
    {                                                                                                                    \
        const cudaError_t err_ = (expr);                                                                                 \
        if (err_ != cudaSuccess)                                                                                         \
        {                                                                                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n  %s\n", cudaGetErrorString(err_), __FILE__, __LINE__, #expr); \
            std::exit(1);                                                                                                \
        }                                                                                                                \
    } while (0)

using Clock = std::chrono::steady_clock;
static double elapsed_ns(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::nano>(b - a).count();
}

__global__ void empty_kernel() {}

struct Stats
{
    double mean, p50, p99, minimum, maximum;
};
// 与原基准 summarize() 完全一致的统计口径
static Stats summarize(std::vector<double> v)
{
    if (v.empty())
        throw std::invalid_argument("empty samples");
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::sort(v.begin(), v.end());
    auto pct = [&](double p)
    { return v[static_cast<size_t>(p * (v.size() - 1))]; };
    return {mean, pct(0.50), pct(0.99), v.front(), v.back()};
}

static void report(const char *id, const char *label, const std::vector<double> &s)
{
    const Stats st = summarize(s);
    std::printf("%-6s%-42s mean=%9.1f ns (%8.3f us) p50=%9.1f p99=%9.1f min=%9.1f max=%11.1f [n=%zu]\n",
                id,
                label,
                st.mean,
                st.mean / 1000.0,
                st.p50,
                st.p99,
                st.minimum,
                st.maximum,
                s.size());
}

int main(int argc, char **argv)
{
    int                      iters = 10000, warmup = 2000, device = 0;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc)
            device = std::atoi(argv[++i]);
        else if (a == "--help")
        {
            std::printf("用法: %s [iters] [warmup] [--device N]\n", argv[0]);
            return 0;
        }
        else
            pos.push_back(a);
    }
    if (pos.size() > 0)
        iters = std::atoi(pos[0].c_str());
    if (pos.size() > 1)
        warmup = std::atoi(pos[1].c_str());
    if (iters < 1 || warmup < 1)
    {
        std::fprintf(stderr, "iters/warmup 必须 >= 1\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int drv = 0, rt = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    std::printf("GPU: %s | device=%d | CC=%d.%d | SMs=%d\n",
                prop.name,
                device,
                prop.major,
                prop.minor,
                prop.multiProcessorCount);
    std::printf("CUDA: compiled=%d runtime=%d driver_API=%d\n", CUDART_VERSION, rt, drv);
    std::printf("iters=%d warmup=%d\n", iters, warmup);
    std::printf("wall = 整轮墙钟；event = GPU 时间轴区间；host = wall - event（主机侧）\n");
    std::printf("单次往返延迟，非稳态吞吐，不能与背靠背发射的摊薄成本直接比较。\n\n");

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    for (int i = 0; i < warmup; ++i)
        empty_kernel<<<1, 1, 0, s>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<double> ev(iters), wall(iters), host(iters);
    for (int i = 0; i < iters; ++i)
    {
        const auto t0 = Clock::now();
        CUDA_CHECK(cudaEventRecord(begin, s));
        empty_kernel<<<1, 1, 0, s>>>();
        CUDA_CHECK(cudaEventRecord(end, s));
        CUDA_CHECK(cudaEventSynchronize(end));
        const auto t1 = Clock::now();
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        ev[i]   = static_cast<double>(ms) * 1e6;
        wall[i] = elapsed_ns(t0, t1);
        host[i] = wall[i] - ev[i];
    }

    report("wall", "整轮墙钟（提交+执行+完成通知）", wall);
    report("event", "GPU 时间轴区间（begin -> end）", ev);
    report("host", "主机侧开销（wall - event）", host);

    const Stats w = summarize(wall), e = summarize(ev), h = summarize(host);
    std::printf("\n按 p50 拆分：整轮 %.1f ns = GPU 时间轴 %.1f (%.0f%%) + 主机侧 %.1f (%.0f%%)\n",
                w.p50,
                e.p50,
                100.0 * e.p50 / w.p50,
                h.p50,
                100.0 * h.p50 / w.p50);

    // ---------- 第二部分：纯 CPU 发射成本（按 N 扫描）----------
    std::printf("\n");
    std::printf("=== 纯 CPU 发射成本：按批量 N 扫描 ===\n");
    std::printf("issue = 只发射不等待的每次成本；drain = 发射完到排空的时间\n");
    std::printf("N 小时 issue 是纯 CPU 成本；N 大到队列填满后，issue 退化为 GPU 排空速率\n\n");
    std::printf("%10s %14s %16s %16s\n", "N", "issue/次(ns)", "drain 总计(ns)", "总计/次(ns)");

    const int sweep[]     = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536};
    double    issue_small = 0.0;
    for (size_t k = 0; k < sizeof(sweep) / sizeof(sweep[0]); ++k)
    {
        const int n    = sweep[k];
        int       reps = 100000 / n;
        if (reps < 20)
            reps = 20;
        if (reps > 2000)
            reps = 2000;
        std::vector<double> iss(reps), dr(reps), tot(reps);
        for (int rep = 0; rep < reps; ++rep)
        {
            CUDA_CHECK(cudaStreamSynchronize(s)); // 先清空此前的工作
            const auto t0 = Clock::now();
            for (int i = 0; i < n; ++i)
                empty_kernel<<<1, 1, 0, s>>>();
            const auto t1 = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(s));
            const auto t2 = Clock::now();
            CUDA_CHECK(cudaGetLastError());
            iss[rep] = elapsed_ns(t0, t1) / n;
            dr[rep]  = elapsed_ns(t1, t2);
            tot[rep] = elapsed_ns(t0, t2) / n;
        }
        const Stats si = summarize(iss), sd = summarize(dr), st2 = summarize(tot);
        std::printf("%10d %14.1f %16.1f %16.1f\n", n, si.p50, sd.p50, st2.p50);
        if (n == 16)
            issue_small = si.p50;
    }
    std::printf("\n小批量（N=16）下每次发射的纯 CPU 成本约 %.1f ns\n", issue_small);
    std::printf("若 issue 随 N 增大而明显上升，说明已撞到队列背压，该列不再代表 CPU 成本\n");

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
