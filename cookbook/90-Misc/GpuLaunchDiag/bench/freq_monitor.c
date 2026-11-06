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

// 测量一个程序运行期间 CPU 的【实际】平均频率，并输出时间序列。
//
// 原理：直接用 perf_event_open 读硬件 PMU 的 CPU_CYCLES 计数器，
//       频率 = 周期数 / 墙上时间。与 /proc/cpuinfo、cpufreq、MSR、TSC 都无关，
//       因此在虚拟机里同样有效（前提是 guest 有可用的 vPMU）。
//
// 前提：perf_event_paranoid <= 3，或进程持有 CAP_PERFMON。
//       （能用 perf_bugs 拿到指令数的机器，这里就一定能用。）
//
// 用法：  freq_monitor [-i 间隔ms] -- <要运行的命令> [参数...]
// 建议：  taskset -c <核> freq_monitor -- taskset -c <核> <命令>
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int g_excl_kernel = 0; // 实际生效的模式，汇总里会说明

static int open_one(unsigned type, unsigned long long config, int excl)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.type           = type;
    a.size           = sizeof a;
    a.config         = config;
    a.disabled       = 1;
    a.inherit        = 1; // inherit: 统计 fork 出来的子进程
    a.exclude_kernel = excl;
    a.exclude_hv     = excl;
    return syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
}
// 先试含内核态；被 paranoid 拒绝则退回只统计用户态（无特权时的常规路径）
static int open_ctr(unsigned long long config)
{
    int fd = open_one(PERF_TYPE_HARDWARE, config, 0);
    if (fd >= 0)
        return fd;
    fd = open_one(PERF_TYPE_HARDWARE, config, 1);
    if (fd >= 0)
        g_excl_kernel = 1;
    return fd;
}
static long long readc(int fd)
{
    long long v = 0;
    if (fd >= 0 && read(fd, &v, sizeof v) != sizeof v)
        v = -1;
    return v;
}

int main(int argc, char **argv)
{
    double gap = 0.2;
    int    i   = 1;
    while (i < argc && strcmp(argv[i], "--"))
    {
        if (!strcmp(argv[i], "-i") && i + 1 < argc)
        {
            gap = atof(argv[++i]) / 1000.0;
        }
        i++;
    }
    if (i >= argc || strcmp(argv[i], "--"))
    {
        fprintf(stderr, "用法: %s [-i 间隔ms] -- <命令> [参数...]\n", argv[0]);
        return 2;
    }
    i++;
    if (i >= argc)
    {
        fprintf(stderr, "缺少要运行的命令\n");
        return 2;
    }

    int fc = open_ctr(PERF_COUNT_HW_CPU_CYCLES);
    int fi = open_ctr(PERF_COUNT_HW_INSTRUCTIONS);
    // task-clock 是软件事件，任何权限下都能开；用它拿到真实占用的 CPU 时间，
    // 比墙上时间更准（排除了进程没在跑的那部分）
    int ft = open_one(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 0);
    if (fc < 0)
    {
        fprintf(stderr, "❌ 打不开 CPU_CYCLES 计数器: %s\n", strerror(errno));
        fprintf(stderr, "   EACCES → perf_event_paranoid 太高，或容器缺 CAP_PERFMON\n");
        fprintf(stderr, "   ENOENT → 本机/本虚拟机没有可用的 PMU（KVM 需要宿主机开 vPMU）\n");
        return 3;
    }
    ioctl(fc, PERF_EVENT_IOC_RESET, 0);
    ioctl(fc, PERF_EVENT_IOC_ENABLE, 0);
    if (fi >= 0)
    {
        ioctl(fi, PERF_EVENT_IOC_RESET, 0);
        ioctl(fi, PERF_EVENT_IOC_ENABLE, 0);
    }
    if (ft >= 0)
    {
        ioctl(ft, PERF_EVENT_IOC_RESET, 0);
        ioctl(ft, PERF_EVENT_IOC_ENABLE, 0);
    }

    double t0  = now();
    pid_t  pid = fork();
    if (pid == 0)
    {
        execvp(argv[i], &argv[i]);
        perror("execvp");
        _exit(127);
    }

    printf("# 相对时间(s)  区间频率(GHz)  累计频率(GHz)\n");
    long long pc = 0;
    double    pt = t0;
    int       st;
    int       done = 0;
    double    lo = 1e9, hi = 0;
    // 轮询用小步长（子进程退出时刻才能精确记到），打印仍按 gap 的节奏。
    // 否则墙上时间会被量化到 gap 上，差分求每次耗时会严重失真。
    const double POLL = 0.002;
    while (!done)
    {
        double seg = 0;
        while (seg < gap)
        {
            struct timespec ts = {0, (long)(POLL * 1e9)};
            nanosleep(&ts, NULL);
            seg += POLL;
            if (waitpid(pid, &st, WNOHANG) == pid)
            {
                done = 1;
                break;
            }
        }
        double    t = now();
        long long c = readc(fc);
        if (c < 0)
            break;
        double dg = (double)(c - pc) / (t - pt) / 1e9, ag = (double)c / (t - t0) / 1e9;
        if (done || t - pt > gap * 0.5)
        {
            printf("  %8.2f      %7.3f       %7.3f\n", t - t0, dg, ag);
            if (dg > hi)
                hi = dg;
            if (dg < lo)
                lo = dg;
        }
        pc = c;
        pt = t;
    }
    if (!done)
        waitpid(pid, &st, 0);
    double    el = now() - t0;
    long long c = readc(fc), ins = readc(fi), tc = readc(ft);
    ioctl(fc, PERF_EVENT_IOC_DISABLE, 0);

    printf("\n================= 汇总 =================\n");
    printf("统计范围   : %s\n", g_excl_kernel ? "仅用户态（paranoid 限制，已自动回退）" : "用户态+内核态");
    printf("墙上时间   : %.3f s\n", el);
    printf("总周期数   : %lld\n", c);
    if (tc > 0)
    {
        double cpu_s = (double)tc / 1e9;
        printf("CPU 占用   : %.3f s  (占墙上时间 %.1f%%)\n", cpu_s, cpu_s / el * 100);
        printf("★平均频率★: %.3f GHz   ← 周期数 / CPU占用时间（推荐看这个）\n", (double)c / cpu_s / 1e9);
        printf("  参考      : %.3f GHz   （周期数 / 墙上时间）\n", (double)c / el / 1e9);
    }
    else
    {
        printf("★平均频率★: %.3f GHz\n", (double)c / el / 1e9);
    }
    if (hi > 0)
        printf("区间频率   : 最低 %.3f / 最高 %.3f GHz\n", lo, hi);
    if (ins > 0)
    {
        printf("总指令数   : %lld\n", ins);
        printf("IPC        : %.3f\n", (double)ins / (double)c);
    }
    printf("========================================\n");
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
