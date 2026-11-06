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

// Standalone CUDA launch benchmark. Build with nvcc, not g++/clang++.
// L20: nvcc -O3 -std=c++14 -arch=sm_89 kernel_launch_bench_split.cu -o kernel_launch_bench_split
// RTX PRO 5000 Blackwell: use CUDA >= 12.8 and -arch=sm_120.
// Run: ./kernel_launch_bench_split 100000 2000 --csv results.csv
// These are API/latency/throughput measurements, NOT hardware-stage timestamps.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if CUDART_VERSION < 11080
    #error "Use CUDA Toolkit 11.8 or newer (12.8+ for native sm_120 compilation)."
#endif

using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady, "A monotonic host clock is required");

static void check_cuda(cudaError_t error, const char *expression, const char *file, int line)
{
    if (error == cudaSuccess)
        return;
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": "
            << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ')';
    throw std::runtime_error(message.str());
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

static void cleanup_check(cudaError_t error, const char *operation) noexcept
{
    if (error != cudaSuccess)
        std::fprintf(stderr, "Cleanup warning: %s: %s\n", operation, cudaGetErrorString(error));
}

struct Stream
{
    cudaStream_t handle = nullptr;
    Stream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
    }
    ~Stream()
    {
        if (handle)
            cleanup_check(cudaStreamDestroy(handle), "stream destroy");
    }
    Stream(const Stream &)            = delete;
    Stream &operator=(const Stream &) = delete;
};

struct Event
{
    cudaEvent_t handle = nullptr;
    Event()
    {
        CUDA_CHECK(cudaEventCreate(&handle));
    }
    ~Event()
    {
        if (handle)
            cleanup_check(cudaEventDestroy(handle), "event destroy");
    }
    Event(const Event &)            = delete;
    Event &operator=(const Event &) = delete;
};

__global__ void empty_kernel() {}
__global__ void empty_kernel_smem() {}
struct Param256
{
    int x[64];
};
struct Param1024
{
    int x[256];
};
static_assert(sizeof(Param256) == 256, "Unexpected parameter size");
static_assert(sizeof(Param1024) == 1024, "Unexpected parameter size");
__global__ void kernel_param_256(Param256 p)
{
    // Keep endpoint loads; this is not a full parameter-memory bandwidth test.
    asm volatile("" ::"r"(p.x[0]), "r"(p.x[63]));
}
__global__ void kernel_param_1024(Param1024 p)
{
    asm volatile("" ::"r"(p.x[0]), "r"(p.x[255]));
}

struct Graph
{
    cudaGraph_t     graph           = nullptr;
    cudaGraphExec_t exec            = nullptr;
    Graph()                         = default;
    Graph(const Graph &)            = delete;
    Graph &operator=(const Graph &) = delete;
    ~Graph()
    {
        if (exec)
            cleanup_check(cudaGraphExecDestroy(exec), "graph exec destroy");
        if (graph)
            cleanup_check(cudaGraphDestroy(graph), "graph destroy");
    }
    void build(cudaStream_t stream, int nodes)
    {
        // Explicit nodes avoid stream-capture state being stranded on an error.
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        for (int i = 0; i < nodes; ++i)
        {
            cudaKernelNodeParams params {};
            params.func             = reinterpret_cast<void *>(empty_kernel);
            params.gridDim          = dim3(1);
            params.blockDim         = dim3(1);
            cudaGraphNode_t current = nullptr;
            CUDA_CHECK(cudaGraphAddKernelNode(&current, graph, i ? &previous : nullptr, i ? 1 : 0, &params));
            previous = current;
        }
#if CUDART_VERSION >= 12000
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0ULL));
#else
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
#endif
        // Upload and first replay are explicitly outside every timed interval.
        CUDA_CHECK(cudaGraphUpload(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

struct Options
{
    int                  iters        = 100000;
    int                  warmup       = 2000;
    int                  rounds       = 3;
    int                  device       = 0;
    bool                 help         = false;
    bool                 skip_cluster = false;
    std::array<bool, 11> tests {};
    std::string          csv;
    Options()
    {
        tests.fill(true);
        tests[0] = false;
    }
};

static int parse_integer(const std::string &text, const char *name, int minimum)
{
    if (text.empty())
        throw std::invalid_argument(std::string(name) + " is empty");
    for (char c : text)
        if (c < '0' || c > '9')
            throw std::invalid_argument(std::string(name) + " must be an integer: " + text);
    errno            = 0;
    char      *end   = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || *end || value < minimum || value > INT_MAX)
        throw std::invalid_argument(std::string(name) + " is out of range: " + text);
    return static_cast<int>(value);
}

static Options parse_options(int argc, char **argv)
{
    Options options;
    int     positional = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg   = argv[i];
        auto              value = [&]() -> std::string
        {
            if (++i >= argc)
                throw std::invalid_argument("Missing value for " + arg);
            return argv[i];
        };
        if (arg == "--help" || arg == "-h")
            options.help = true;
        else if (arg == "--skip-cluster")
            options.skip_cluster = true;
        else if (arg == "--iters")
            options.iters = parse_integer(value(), "iters", 1);
        else if (arg == "--warmup")
            options.warmup = parse_integer(value(), "warmup", 1);
        else if (arg == "--rounds")
            options.rounds = parse_integer(value(), "rounds", 1);
        else if (arg == "--device")
            options.device = parse_integer(value(), "device", 0);
        else if (arg == "--csv")
        {
            options.csv = value();
            if (options.csv.empty())
                throw std::invalid_argument("CSV path is empty");
        }
        else if (arg == "--tests")
        {
            const std::string selected = value();
            options.tests.fill(selected == "all");
            options.tests[0] = false;
            if (selected == "all")
                continue;
            if (selected.empty() || selected.back() == ',')
                throw std::invalid_argument("--tests expects all or a comma-separated list, e.g. 1,2,9");
            std::istringstream input(selected);
            std::string        token;
            while (std::getline(input, token, ','))
            {
                const int test = parse_integer(token, "test number", 1);
                if (test > 10)
                    throw std::invalid_argument("Test numbers must be 1..10");
                options.tests[test] = true;
            }
        }
        else if (!arg.empty() && arg[0] != '-' && positional < 2)
        {
            const int number = parse_integer(arg, positional ? "warmup" : "iters", 1);
            if (positional++ == 0)
                options.iters = number;
            else
                options.warmup = number;
        }
        else
            throw std::invalid_argument("Unknown argument: " + arg);
    }
    return options;
}

static void print_help(const char *program)
{
    std::printf("Usage: %s [iters=100000] [warmup=2000] [options]\n"
                "  --iters N         Samples/launches per test (N >= 1)\n"
                "  --warmup N        Untimed warmup launches (N >= 1)\n"
                "  --rounds N        Repetitions of throughput tests (default 3)\n"
                "  --device N        CUDA-visible device index (default 0)\n"
                "  --tests LIST      all or e.g. 1,2,4,9\n"
                "  --skip-cluster    Skip optional test 6\n"
                "  --csv PATH        Write results, metadata and SKIP rows\n"
                "  --help            Show this help without initializing CUDA\n",
                program);
}

struct Stats
{
    double mean, p50, p99, minimum, maximum;
};
static Stats summarize(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("Cannot summarize empty samples");
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p)
    { return values[static_cast<size_t>(p * (values.size() - 1))]; };
    return {mean, percentile(0.50), percentile(0.99), values.front(), values.back()};
}
static std::string quote_csv(const std::string &value)
{
    std::string escaped = "\"";
    for (char c : value)
    {
        if (c == '"')
            escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

class Reporter
{
    std::ofstream csv_;
    std::string   gpu_;
    int           driver_, runtime_;
    Options       options_;

public:
    Reporter(const Options &options, const char *gpu, int driver, int runtime):
        gpu_(gpu), driver_(driver), runtime_(runtime), options_(options)
    {
        if (options.csv.empty())
            return;
        csv_.open(options.csv);
        if (!csv_)
            throw std::runtime_error("Cannot open CSV: " + options.csv);
        csv_ << "gpu,driver_version,runtime_version,iters,warmup,rounds,test,status,metric,unit,samples,"
                "mean_ns,p50_ns,p99_ns,min_ns,max_ns,note\n";
    }
    void record(const std::string &id, const std::string &metric, const std::vector<double> &samples, const std::string &unit, const std::string &note = "")
    {
        const Stats stats = summarize(samples);
        std::printf("%-5s %-44s mean=%9.1f ns (%8.3f us) p50=%9.1f p99=%9.1f min=%9.1f max=%10.1f [%s; n=%zu]\n",
                    id.c_str(),
                    metric.c_str(),
                    stats.mean,
                    stats.mean / 1000.0,
                    stats.p50,
                    stats.p99,
                    stats.minimum,
                    stats.maximum,
                    unit.c_str(),
                    samples.size());
        if (!csv_.is_open())
            return;
        prefix(id, "OK", metric, unit);
        csv_ << samples.size() << ',' << std::fixed << std::setprecision(3)
             << stats.mean << ',' << stats.p50 << ',' << stats.p99 << ','
             << stats.minimum << ',' << stats.maximum << ',' << quote_csv(note) << '\n';
        flush();
    }
    void skip(const std::string &id, const std::string &reason)
    {
        std::printf("%-5s SKIP: %s\n", id.c_str(), reason.c_str());
        if (!csv_.is_open())
            return;
        prefix(id, "SKIP", "", "");
        csv_ << "0,,,,,," << quote_csv(reason) << '\n';
        flush();
    }

private:
    void prefix(const std::string &id, const char *status, const std::string &metric, const std::string &unit)
    {
        csv_ << quote_csv(gpu_) << ',' << driver_ << ',' << runtime_ << ','
             << options_.iters << ',' << options_.warmup << ',' << options_.rounds << ','
             << quote_csv(id) << ',' << status << ',' << quote_csv(metric) << ','
             << quote_csv(unit) << ',';
    }
    void flush()
    {
        csv_.flush();
        if (!csv_)
            throw std::runtime_error("Writing CSV failed");
    }
};

static double elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::nano>(end - begin).count();
}

// Preflight and final checks are outside call-only timed intervals. A queued
// launch can still block due to backpressure; this is not pure CPU instruction cost.
template<class Launch, class Sync>
static void warmup(int count, Launch launch, Sync sync)
{
    CUDA_CHECK(cudaGetLastError());
    launch();
    CUDA_CHECK(cudaGetLastError());
    sync();
    for (int i = 1; i < count; ++i)
        launch();
    CUDA_CHECK(cudaGetLastError());
    sync();
}

template<class Launch, class Sync>
static std::vector<double> measure_calls(const Options &options, Launch launch, Sync sync, bool sync_each)
{
    warmup(options.warmup, launch, sync);
    std::vector<double> samples(options.iters);
    for (int i = 0; i < options.iters; ++i)
    {
        const auto begin = Clock::now();
        launch();
        if (sync_each)
            sync();
        const auto end = Clock::now();
        samples[i]     = elapsed_ns(begin, end);
    }
    CUDA_CHECK(cudaGetLastError());
    sync();
    return samples;
}

template<class Launch, class Sync>
static std::vector<double> measure_throughput(const Options &options,
                                              Launch         launch,
                                              Sync           sync)
{
    warmup(options.warmup, launch, sync);
    std::vector<double> samples(options.rounds);
    for (int round = 0; round < options.rounds; ++round)
    {
        const auto begin = Clock::now();
        for (int i = 0; i < options.iters; ++i)
            launch();
        sync(); // Completion IS included: an end-to-end throughput measurement.
        const auto end = Clock::now();
        CUDA_CHECK(cudaGetLastError());
        samples[round] = elapsed_ns(begin, end) / options.iters;
    }
    return samples;
}

static void bench_basic(const Options &o, Reporter &r, cudaStream_t s)
{
    auto launch = [&]
    { empty_kernel<<<1, 1, 0, s>>>(); };
    auto sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    if (o.tests[1])
    {
        auto device_sync = []
        { CUDA_CHECK(cudaDeviceSynchronize()); };
        r.record("1", "Launch + device sync", measure_calls(o, launch, device_sync, true), "per launch");
        r.record("1b", "Launch + stream sync", measure_calls(o, launch, sync, true), "per launch", "Matched synchronization scope for test 4a");
    }
    if (o.tests[2])
        r.record("2", "Launch API only", measure_calls(o, launch, sync, false), "per call");
    if (o.tests[3])
        r.record("3", "Back-to-back launch + final sync", measure_throughput(o, launch, sync), "per kernel; round averages");
}

static void bench_graph(const Options &o, Reporter &r, cudaStream_t s)
{
    Graph graph;
    graph.build(s, 1);
    auto launch = [&]
    { CUDA_CHECK(cudaGraphLaunch(graph.exec, s)); };
    auto sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    r.record("4a", "1-node graph replay + stream sync", measure_calls(o, launch, sync, true), "per replay");
    r.record("4b", "1-node graph API only", measure_calls(o, launch, sync, false), "per call");
    r.record("4c", "Back-to-back replay + final sync", measure_throughput(o, launch, sync), "per replay; round averages");
}

static void bench_events(const Options &o, Reporter &r, cudaStream_t s)
{
    Event begin, end;
    auto  launch = [&]
    { empty_kernel<<<1, 1, 0, s>>>(); };
    auto sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    warmup(o.warmup, launch, sync);
    std::vector<double> samples(std::min(o.iters, 10000));
    for (double &sample : samples)
    {
        CUDA_CHECK(cudaEventRecord(begin.handle, s));
        launch();
        CUDA_CHECK(cudaEventRecord(end.handle, s));
        CUDA_CHECK(cudaEventSynchronize(end.handle));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin.handle, end.handle));
        sample = static_cast<double>(ms) * 1e6;
    }
    r.record("5", "GPU event interval around one launch", samples, "per event interval", "Includes GPU timeline gaps; not isolated kernel instruction time; finite event resolution");
}

static bool optional_unsupported(cudaError_t error)
{
    return error == cudaErrorNotSupported || error == cudaErrorInvalidValue || error == cudaErrorInvalidConfiguration || error == cudaErrorLaunchOutOfResources;
}

static void bench_cluster(const Options &o, Reporter &r, cudaStream_t s)
{
    if (o.skip_cluster)
    {
        r.skip("6", "Disabled by --skip-cluster");
        return;
    }
#if CUDART_VERSION >= 11080
    int supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, o.device));
    if (!supported)
    {
        r.skip("6", "Device does not support cluster launch");
        return;
    }
    cudaLaunchConfig_t regular {};
    regular.gridDim               = dim3(8);
    regular.blockDim              = dim3(32);
    regular.stream                = s;
    int               max_cluster = 0;
    const cudaError_t occupancy   = cudaOccupancyMaxPotentialClusterSize(&max_cluster, empty_kernel, &regular);
    if (optional_unsupported(occupancy))
    {
        r.skip("6", std::string("Cluster capability query: ") + cudaGetErrorString(occupancy));
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(occupancy);
    if (max_cluster < 2)
    {
        r.skip("6", "A 2-block cluster is unavailable for this configuration");
        return;
    }
    cudaLaunchAttribute attribute {};
    attribute.id                 = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x   = 2;
    attribute.val.clusterDim.y   = 1;
    attribute.val.clusterDim.z   = 1;
    cudaLaunchConfig_t clustered = regular;
    clustered.attrs              = &attribute;
    clustered.numAttrs           = 1;
    // Probe optional configuration once before any samples are reported.
    const cudaError_t probe = cudaLaunchKernelEx(&clustered, empty_kernel);
    if (optional_unsupported(probe))
    {
        r.skip("6", std::string("Cluster preflight: ") + cudaGetErrorString(probe));
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(probe);
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    auto cluster_launch = [&]
    { CUDA_CHECK(cudaLaunchKernelEx(&clustered, empty_kernel)); };
    auto regular_ex = [&]
    { CUDA_CHECK(cudaLaunchKernelEx(&regular, empty_kernel)); };
    auto regular_launch = [&]
    { empty_kernel<<<8, 32, 0, s>>>(); };
    const auto cluster_calls = measure_calls(o, cluster_launch, sync, false);
    r.record("6a", "Cluster Ex API (8x32, cluster=2)", cluster_calls, "per call");
    r.record("6b", "Cluster Ex + stream sync", measure_calls(o, cluster_launch, sync, true), "per launch");
    r.record("6c", "Regular <<<8,32>>> API", measure_calls(o, regular_launch, sync, false), "per call");
    const auto regular_calls = measure_calls(o, regular_ex, sync, false);
    r.record("6d", "Regular Ex API (same API, no cluster)", regular_calls, "per call");
    r.record("6e", "Regular Ex + stream sync", measure_calls(o, regular_ex, sync, true), "per launch");
    const double baseline = summarize(regular_calls).mean;
    if (baseline > 0.0)
        std::printf("      Cluster / regular Ex API ratio: %.3fx (6a / 6d)\n",
                    summarize(cluster_calls).mean / baseline);
#else
    r.skip("6", "Toolkit does not expose cluster launch APIs");
#endif
}

static void bench_params(const Options &o, Reporter &r, cudaStream_t s)
{
    Param256  p256 {};
    Param1024 p1024 {};
    p256.x[0] = p1024.x[0] = 1;
    p256.x[63] = p1024.x[255] = 2;
    auto sync                 = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    auto a = [&]
    { empty_kernel<<<1, 1, 0, s>>>(); };
    auto b = [&]
    { kernel_param_256<<<1, 1, 0, s>>>(p256); };
    auto c = [&]
    { kernel_param_1024<<<1, 1, 0, s>>>(p1024); };
    r.record("7a", "No user parameters: API", measure_calls(o, a, sync, false), "per call");
    r.record("7b", "256-byte parameters: API", measure_calls(o, b, sync, false), "per call");
    r.record("7c", "1024-byte parameters: API", measure_calls(o, c, sync, false), "per call");
    r.record("7a.t", "No user parameters: throughput", measure_throughput(o, a, sync), "per kernel; round averages");
    r.record("7b.t", "256-byte parameters: throughput", measure_throughput(o, b, sync), "per kernel; round averages");
    r.record("7c.t", "1024-byte parameters: throughput", measure_throughput(o, c, sync), "per kernel; round averages");
}

static void bench_streams(const Options &o, Reporter &r)
{
    std::array<Stream, 8> streams;
    auto                  one = [&]
    { empty_kernel<<<1, 1, 0, streams[0].handle>>>(); };
    auto one_sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(streams[0].handle)); };
    r.record("8a", "1 stream: launch + final sync", measure_throughput(o, one, one_sync), "per kernel; round averages");
    unsigned index = 0;
    auto     many  = [&]
    {
        const cudaStream_t s = streams[index].handle;
        index                = (index + 1) % streams.size();
        empty_kernel<<<1, 1, 0, s>>>();
    };
    auto many_sync = [&]
    {
        for (auto &stream : streams)
            CUDA_CHECK(cudaStreamSynchronize(stream.handle));
        index = 0;
    };
    // Preflight every stream even when --warmup is less than the stream count.
    for (auto &stream : streams)
        empty_kernel<<<1, 1, 0, stream.handle>>>();
    CUDA_CHECK(cudaGetLastError());
    many_sync();
    r.record("8b", "8 streams: launch + final sync", measure_throughput(o, many, many_sync), "per kernel; round averages");
}

static void bench_graph_nodes(const Options &o, Reporter &r, cudaStream_t s)
{
    for (int nodes : {1, 8, 32})
    {
        Graph graph;
        graph.build(s, nodes);
        auto launch = [&]
        { CUDA_CHECK(cudaGraphLaunch(graph.exec, s)); };
        auto sync = [&]
        { CUDA_CHECK(cudaStreamSynchronize(s)); };
        const std::string name = "Graph nodes=" + std::to_string(nodes);
        r.record("9a", name + ": API only", measure_calls(o, launch, sync, false), "per call");
        const auto replay = measure_throughput(o, launch, sync);
        r.record("9b", name + ": replay + final sync", replay, "per replay; round averages");
        auto per_kernel = replay;
        for (double &value : per_kernel)
            value /= nodes;
        r.record("9c", name + ": amortized per kernel", per_kernel, "per kernel; round averages", "Derived from 9b / nodes; not individual kernel latency or node-gap duration");
    }
}

static void bench_shapes(const Options &o, Reporter &r, cudaStream_t s)
{
    auto sync = [&]
    { CUDA_CHECK(cudaStreamSynchronize(s)); };
    auto run = [&](const char *id, const char *label, int grid, int block, size_t smem)
    {
        auto launch = [&]
        { empty_kernel_smem<<<grid, block, smem, s>>>(); };
        r.record(id, label, measure_calls(o, launch, sync, false), "per call");
        r.record(std::string(id) + ".t", std::string(label) + " throughput", measure_throughput(o, launch, sync), "per kernel; round averages");
    };
    run("10a", "Shape <<<1,1,0>>> API", 1, 1, 0);
    run("10b", "Shape <<<8,32,0>>> API", 8, 32, 0);
    // Extra controls isolate block count from threads per block.
    run("10d", "Shape <<<1,32,0>>> API", 1, 32, 0);
    run("10e", "Shape <<<8,1,0>>> API", 8, 1, 0);
    constexpr int      requested = 48 * 1024;
    cudaFuncAttributes attributes {};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(empty_kernel_smem)));
    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, o.device));
    if (static_cast<size_t>(requested) + attributes.sharedSizeBytes > static_cast<size_t>(max_optin))
    {
        r.skip("10c", "48 KiB dynamic shared memory exceeds device limit");
        return;
    }
    if (attributes.maxDynamicSharedSizeBytes < requested)
    {
        const cudaError_t status = cudaFuncSetAttribute(reinterpret_cast<const void *>(empty_kernel_smem),
                                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                        requested);
        if (optional_unsupported(status))
        {
            r.skip("10c", std::string("Shared-memory opt-in: ") + cudaGetErrorString(status));
            (void)cudaGetLastError();
            return;
        }
        CUDA_CHECK(status);
    }
    run("10c", "Shape <<<8,32,48KiB>>> API", 8, 32, requested);
}

int main(int argc, char **argv)
{
    // Preserve diagnostics if stdout is redirected to a log.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try
    {
        const Options options = parse_options(argc, argv);
        if (options.help)
        {
            print_help(argv[0]);
            return 0;
        }
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count == 0)
            throw std::runtime_error("No CUDA device is visible");
        if (options.device >= count)
            throw std::invalid_argument("--device is outside the visible device range");
        CUDA_CHECK(cudaSetDevice(options.device));
        // Initialize runtime/context before gathering any timing samples.
        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp prop {};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
        int driver = 0, runtime = 0, clock_khz = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, options.device));
        std::printf("GPU: %s | device=%d | CC=%d.%d | SMs=%d\n"
                    "CUDA: compiled=%d runtime=%d driver_API=%d\n"
                    "Reported clock attribute: %.0f MHz (not a measured/locked clock)\n"
                    "iters=%d warmup=%d throughput_rounds=%d\n",
                    prop.name,
                    options.device,
                    prop.major,
                    prop.minor,
                    prop.multiProcessorCount,
                    CUDART_VERSION,
                    runtime,
                    driver,
                    clock_khz / 1000.0,
                    options.iters,
                    options.warmup,
                    options.rounds);
        std::printf("All units are ns; 1000 ns = 1 us. Throughput percentiles describe round averages.\n"
                    "Call-only: API return time, possibly including backpressure.\n"
                    "Final-sync: end-to-end throughput, NOT pure CPU time.\n"
                    "Event: elapsed GPU timeline interval, NOT isolated SM instruction time.\n"
                    "No result isolates doorbell/I2M/QMD/semaphore hardware stages.\n\n");
        const char *blocking = std::getenv("CUDA_LAUNCH_BLOCKING");
        if (blocking && std::string(blocking) != "0" && *blocking)
            throw std::runtime_error("Unset CUDA_LAUNCH_BLOCKING for this asynchronous benchmark");
        Stream stream;
        // Fail early if the binary has no kernel image for the chosen GPU.
        empty_kernel<<<1, 1, 0, stream.handle>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.handle));
        Reporter reporter(options, prop.name, driver, runtime);
        // Quantify timer-pair overhead; do not subtract noisy calibration values.
        std::vector<double> timer_samples(std::min(options.iters, 10000));
        for (double &sample : timer_samples)
        {
            const auto begin = Clock::now();
            const auto end   = Clock::now();
            sample           = elapsed_ns(begin, end);
        }
        reporter.record("timer", "Back-to-back host clock reads", timer_samples, "per timer pair");
        bench_basic(options, reporter, stream.handle);
        if (options.tests[4])
            bench_graph(options, reporter, stream.handle);
        if (options.tests[5])
            bench_events(options, reporter, stream.handle);
        if (options.tests[6])
            bench_cluster(options, reporter, stream.handle);
        if (options.tests[7])
            bench_params(options, reporter, stream.handle);
        if (options.tests[8])
            bench_streams(options, reporter);
        if (options.tests[9])
            bench_graph_nodes(options, reporter, stream.handle);
        if (options.tests[10])
            bench_shapes(options, reporter, stream.handle);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!options.csv.empty())
            std::printf("\nCSV: %s\n", options.csv.c_str());
        std::puts("Done. Unsupported optional cases are reported as SKIP.");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "\nERROR: %s\n", error.what());
        std::fprintf(stderr, "Check --help, nvcc --version, nvidia-smi, and the compiled GPU architecture.\n");
        return 1;
    }
}
