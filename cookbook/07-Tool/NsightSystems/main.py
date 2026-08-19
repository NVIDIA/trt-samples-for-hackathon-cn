# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read an Nsight Systems report from code, instead of opening it in `nsys-ui`.

`main.sh` next door produces two `.nsys-rep` files and never looks at them again: everything they
contain is only reachable by a human with a GUI. `nsys export --type sqlite` turns a report into an
ordinary SQLite database, which makes the timeline queryable - and, more usefully here, makes it
possible to answer "how much GPU time did each TensorRT layer take" without a profiler API.

The cases below build up to that, and stop at the trap that makes a naive reading of the database
wrong by more than an order of magnitude: **TensorRT runs the network as a CUDA graph, and by
default Nsight Systems does not record what is inside a CUDA graph.**
"""

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from tensorrt_cookbook import case_mark, cookbook_path

current_path = Path(__file__).parent
engine_file = current_path / "model-trained.trt"
onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")

N_ITERATION = 50
# One report per CUDA-graph granularity; `graph` is the nsys default, `node` is the fix.
report_dict = {granularity: current_path / f"sqlite-{granularity}" for granularity in ("graph", "node")}

def database_of(granularity: str) -> Path:
    return report_dict[granularity].with_suffix(".sqlite")

def query(granularity: str, sql: str, parameter_tuple=()):
    """Run one query, returning [] if the table the query needs does not exist in this report."""
    with sqlite3.connect(f"file:{database_of(granularity)}?mode=ro", uri=True) as connection:
        try:
            return connection.execute(sql, parameter_tuple).fetchall()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return []
            raise

def table_exists(granularity: str, name: str) -> bool:
    with sqlite3.connect(f"file:{database_of(granularity)}?mode=ro", uri=True) as connection:
        return connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name, )).fetchone()[0] > 0

# NVTX text lives in one of two places, see `case_the_nvtx_text_trap`.
NVTX_NAME = "COALESCE(n.text, s.value)"

@case_mark
def case_profile_and_export():
    """Profile `trtexec` twice, once per CUDA-graph granularity, and export both to SQLite.

    Note what is *not* traced: no CPU sampling, no context switches, only CUDA and NVTX. `main.sh`
    leaves those on and its reports are ~52 MB each; the ones here are ~140 KB, which matters
    because `nsys export` walks every event in the report.
    """
    if not engine_file.exists():
        subprocess.run(["trtexec", f"--onnx={onnx_file}", f"--saveEngine={engine_file}", "--skipInference"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for granularity, report in report_dict.items():
        subprocess.run(
            ["nsys", "profile", "--force-overwrite=true", f"-o{report}", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", f"--cuda-graph-trace={granularity}", "trtexec", f"--loadEngine={engine_file}", f"--iterations={N_ITERATION}", "--warmUp=0", "--duration=0"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        start_time = time.time()
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(database_of(granularity)), str(report.with_suffix(".nsys-rep"))], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        export_time = time.time() - start_time
        report_size = report.with_suffix(".nsys-rep").stat().st_size
        print(f"    --cuda-graph-trace={granularity:5s}: report {report_size / 2**10:7.1f} KiB -> sqlite {database_of(granularity).stat().st_size / 2**10:8.1f} KiB, export {export_time:.1f} s")
    return

@case_mark
def case_survey_the_database():
    """Which of the ~66 tables actually hold anything. Most are empty enum lookups."""
    with sqlite3.connect(f"file:{database_of('graph')}?mode=ro", uri=True) as connection:
        name_list = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        populated = []
        for name in name_list:
            count = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            if count > 0 and not name.startswith("ENUM_"):
                populated.append((name, count))
    print(f"    {len(name_list)} tables, {len(populated)} non-empty and not an ENUM_ lookup. The ones that matter here:")
    for name, count in sorted(populated, key=lambda x: -x[1]):
        if name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_MEMCPY"):
            print(f"        {name:36s} {count:>8d}")
    return

@case_mark
def case_the_nvtx_text_trap():
    """NVTX text is in `NVTX_EVENTS.text` **or** in `StringIds`, never both. Miss it and the
    TensorRT layer names silently vanish from the result set."""
    inline = query("graph", "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text IS NOT NULL")[0][0]
    interned = query("graph", "SELECT COUNT(*) FROM NVTX_EVENTS WHERE textId IS NOT NULL")[0][0]
    both = query("graph", "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text IS NOT NULL AND textId IS NOT NULL")[0][0]
    print(f"    NVTX_EVENTS rows with inline `text`: {inline}, with interned `textId`: {interned}, with both: {both}")

    naive = query("graph", "SELECT COUNT(DISTINCT text) FROM NVTX_EVENTS WHERE text LIKE '%myl%'")[0][0]
    correct = query("graph", f"SELECT COUNT(DISTINCT {NVTX_NAME}) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId WHERE {NVTX_NAME} LIKE '%myl%'")[0][0]
    print(f"    TensorRT layer ranges found by `WHERE text LIKE ...`      : {naive}")
    print(f"    TensorRT layer ranges found by COALESCE(text, StringIds) : {correct}")
    print(f"    -> the registered-string path is where TensorRT puts layer names; always COALESCE")
    assert naive == 0 and correct > 0, "the NVTX text trap no longer reproduces, re-check the query"
    return

@case_mark
def case_tensorrt_layers_from_nvtx():
    """TensorRT annotates each layer with an NVTX range, so the layer names are already in here."""
    row_list = query("graph", f"""SELECT {NVTX_NAME} AS name, COUNT(*), SUM(n.end - n.start)
                     FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
                     WHERE n.end IS NOT NULL GROUP BY name ORDER BY 3 DESC LIMIT 8""")
    print(f"    {'NVTX range':46s} {'count':>6s} {'CPU ns':>10s}")
    for name, count, duration in row_list:
        print(f"    {str(name)[:46]:46s} {count:6d} {duration:10d}")
    print(f"    -> `node_*` are ONNX node names carried through; `__myl_*` are fused Myelin regions")
    print(f"    -> ranges cover the CPU-side enqueue, not the GPU work; case 6 joins them to kernels")
    return

@case_mark
def case_the_cuda_graph_trap():
    """The one that matters: by default the kernel table is nearly empty, and the total is wrong.

    TensorRT executes the network as a CUDA graph. `nsys` defaults to `--cuda-graph-trace=graph`,
    which records each graph *launch* as a single opaque activity and does not record the nodes
    inside it. So the kernel table only ever sees the handful of kernels from the capture pass.
    """
    summary = {}
    for granularity in ("graph", "node"):
        kernel = query(granularity, "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0]
        graph_trace = query(granularity, "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        launch = query(granularity, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId WHERE s.value LIKE 'cudaGraphLaunch%'")[0][0]
        summary[granularity] = dict(
            n_kernel=kernel[0],
            kernel_ns=kernel[1],
            has_graph_table=table_exists(granularity, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"),
            n_graph=graph_trace[0][0] if graph_trace else 0,
            graph_ns=graph_trace[0][1] if graph_trace else 0,
            n_launch=launch,
        )

    for granularity, d in summary.items():
        print(f"    --cuda-graph-trace={granularity}")
        print(f"        CUPTI_ACTIVITY_KIND_KERNEL      : {d['n_kernel']:5d} rows, {d['kernel_ns'] / 1e6:8.3f} ms")
        if d["has_graph_table"]:
            print(f"        CUPTI_ACTIVITY_KIND_GRAPH_TRACE : {d['n_graph']:5d} rows, {d['graph_ns'] / 1e6:8.3f} ms")
        else:
            print(f"        CUPTI_ACTIVITY_KIND_GRAPH_TRACE : table does not exist in this report")
        print(f"        cudaGraphLaunch calls           : {d['n_launch']:5d}")

    graph, node = summary["graph"], summary["node"]
    print(f"\n    Reading GPU time off the kernel table alone:")
    print(f"        default granularity : {graph['kernel_ns'] / 1e6:.3f} ms from {graph['n_kernel']} kernels")
    print(f"        node granularity    : {node['kernel_ns'] / 1e6:.3f} ms from {node['n_kernel']} kernels")
    print(f"        -> the default under-reports GPU time by {node['kernel_ns'] / max(graph['kernel_ns'], 1):.0f}x")
    print(f"    The default is not lying, it is answering a different question:")
    print(f"        graph-mode GRAPH_TRACE total {graph['graph_ns'] / 1e6:.3f} ms vs node-mode kernel total {node['kernel_ns'] / 1e6:.3f} ms")
    print(f"        ({abs(graph['graph_ns'] - node['kernel_ns']) / node['kernel_ns'] * 100:.1f}% apart -- same work, different granularity of recording)")

    n_replay = graph["n_graph"]
    n_captured = graph["n_kernel"]
    print(f"\n    Where {node['n_kernel']} comes from: {n_replay} graph replays x {n_captured} kernels + {n_captured} captured = {n_replay * n_captured + n_captured}")
    assert node["n_kernel"] == n_replay * n_captured + n_captured, "the replay arithmetic no longer holds, re-derive it"
    return

@case_mark
def case_per_layer_gpu_time():
    """The payoff: GPU nanoseconds per TensorRT layer, with no profiler API involved.

    The join is three hops, because NVTX ranges and kernels live on different timelines:
    NVTX range (CPU) contains the launch API call (CPU) -- `correlationId` --> kernel (GPU).
    """
    sql = f"""
        SELECT {NVTX_NAME} AS name, COUNT(k.correlationId), SUM(k.end - k.start)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
             ON r.start >= n.start AND r.end <= n.end AND r.globalTid = n.globalTid
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
        WHERE n.end IS NOT NULL AND {NVTX_NAME} LIKE '%myl0_%'
        GROUP BY name ORDER BY 3 DESC
    """
    row_list = query("node", sql)
    total = sum(row[2] for row in row_list)
    print(f"    {'TensorRT layer':34s} {'kernels':>8s} {'GPU ns':>9s} {'share':>7s}")
    for name, n_kernel, gpu_ns in row_list:
        print(f"    {str(name)[:34]:34s} {n_kernel:8d} {gpu_ns:9d} {gpu_ns / total * 100:6.1f}%")
    print(f"    {'total':34s} {'':8s} {total:9d}")

    enqueue = query("node", f"""SELECT SUM(k.end - k.start) FROM NVTX_EVENTS n
                                LEFT JOIN StringIds s ON s.id = n.textId
                                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.start >= n.start AND r.end <= n.end AND r.globalTid = n.globalTid
                                JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
                                WHERE {NVTX_NAME} = 'ExecutionContext::enqueueV3'""")[0][0]
    print(f"    -> the per-layer times sum to the enclosing `enqueueV3` range exactly ({total} == {enqueue})")
    assert total == enqueue, "per-layer times no longer sum to the enqueue total"
    print(f"    -> BUT this covers only the capture pass: the {N_ITERATION} replayed iterations carry no NVTX,")
    print(f"       so per-layer attribution exists for one iteration, not for the steady state")
    return

@case_mark
def case_cross_check_with_nsys_stats():
    """`nsys stats` answers the common questions without SQL. Use SQL when it does not have a report."""
    # `--force-export=true` is required here: `nsys stats` derives its own `<report>.sqlite`, which
    # collides with the export made in case 1, and it refuses to reuse one it considers stale.
    result = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true", str(report_dict["node"].with_suffix(".nsys-rep"))], capture_output=True, text=True, check=False)
    line_list = [line for line in result.stdout.splitlines() if "," in line and not line.startswith("**")]
    assert len(line_list) > 1, f"nsys stats produced no rows:\n{result.stdout}\n{result.stderr}"

    header, *row_list = line_list
    print(f"    nsys stats --report cuda_gpu_kern_sum, first 4 of {len(row_list)} kernels:")
    print(f"        {'Total ns':>9s} {'Inst':>5s}  Name")
    for row in row_list[:4]:
        field_list = row.split(",")
        print(f"        {field_list[1]:>9s} {field_list[2]:>5s}  {field_list[-1][:64]}")

    # Cross-check against case 5: every kernel ran once per recorded iteration.
    instance_set = {int(row.split(",")[2]) for row in row_list}
    total_ns = sum(int(row.split(",")[1]) for row in row_list)
    print(f"    instance counts across kernels: {sorted(instance_set)} (one per recorded iteration, pooling runs twice)")
    print(f"    total {total_ns / 1e6:.3f} ms -- the same number case 5 got from CUPTI_ACTIVITY_KIND_KERNEL")
    print(f"    -> `nsys stats --help-reports` lists ~40 of these; reach for SQL only when none fits,")
    print(f"       as in case 6, where the join from NVTX layer names to kernels has no built-in report")
    return

if __name__ == "__main__":
    if shutil.which("nsys") is None:
        print("`nsys` not found, skip. Install Nsight Systems (it ships with CUDA at /usr/local/cuda/bin).")
        sys.exit(0)

    case_profile_and_export()
    case_survey_the_database()
    case_the_nvtx_text_trap()
    case_tensorrt_layers_from_nvtx()
    case_the_cuda_graph_trap()
    case_per_layer_gpu_time()
    case_cross_check_with_nsys_stats()

    print("\nFinish")
