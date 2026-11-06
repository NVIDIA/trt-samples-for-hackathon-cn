#!/usr/bin/env bash
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

# 环境采集 —— 硬件 + 软件，一次抓全。
# 专门覆盖「KVM vs 裸金属」「更换 CPU」两类对比需要的项。
# 全程只读，不修改任何配置；有 root 会多抓几项，没有也能跑。
set -u
sec() { echo; echo "========== $* =========="; }
have() { command -v "$1" >/dev/null 2>&1; }
cat_or() { [ -r "$1" ] && cat "$1" 2>/dev/null || echo "(不可读: $1)"; }

sec 基本标识
date -Is; echo "hostname: $(hostname)"; echo "uname: $(uname -a)"
[ -r /etc/os-release ] && grep -E "^(PRETTY_NAME|VERSION)=" /etc/os-release
echo "glibc: $(ldd --version 2>/dev/null | head -1)"
echo "当前用户: $(id -un)  root: $([ "$(id -u)" = 0 ] && echo yes || echo no)"

sec 虚拟化与容器
echo "systemd-detect-virt: $(have systemd-detect-virt && systemd-detect-virt 2>/dev/null || echo 'n/a')"
echo "CPU hypervisor flag: $(grep -qm1 ' hypervisor' /proc/cpuinfo && echo '有 (运行在虚拟机中)' || echo '无 (裸金属)')"
echo "hypervisor 厂商: $(grep -m1 -oE 'hypervisor_vendor.*' /proc/cpuinfo 2>/dev/null || echo 'n/a')"
[ -d /sys/hypervisor ] && { echo "/sys/hypervisor/type: $(cat_or /sys/hypervisor/type)"; }
[ -f /.dockerenv ] && echo "容器: 在 Docker 中" || echo "容器: 未检测到 /.dockerenv"
echo "cgroup: $(head -1 /proc/1/cgroup 2>/dev/null)"
if have dmidecode && [ "$(id -u)" = 0 ]; then
    echo "--- DMI (需 root) ---"
    dmidecode -s system-manufacturer 2>/dev/null
    dmidecode -s system-product-name 2>/dev/null
fi
echo "--- dmesg 中的虚拟化线索（可能需要权限）---"
dmesg 2>/dev/null | grep -iE "kvm|hypervisor|paravirt" | head -5 || echo "(dmesg 不可读)"

sec ★时钟源★
# 这一项对本次对比极重要：VM 下常为 kvm-clock，其 clock_gettime 开销远高于 tsc，
# 会直接抬高基准里的 timer 基线，并影响所有 steady_clock 计时。
echo "current:   $(cat_or /sys/devices/system/clocksource/clocksource0/current_clocksource)"
echo "available: $(cat_or /sys/devices/system/clocksource/clocksource0/available_clocksource)"
echo "TSC 相关 flag:"
for f in constant_tsc nonstop_tsc tsc_reliable rdtscp tsc_known_freq tsc_adjust; do
    grep -qm1 " $f" /proc/cpuinfo && echo "  ✓ $f" || echo "  ✗ $f"
done

sec CPU
have lscpu && lscpu | grep -vE "^Flags|^Vulnerability" || echo "(无 lscpu)"
echo "--- CPU flags（单行，便于 diff）---"
grep -m1 ^flags /proc/cpuinfo | cut -c1-2000
echo "--- 各核当前频率 (MHz) ---"
grep "cpu MHz" /proc/cpuinfo | awk '{printf "%.0f ", $4}'; echo
echo "--- cpufreq ---"
echo "scaling_driver: $(cat_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver)"
echo "governor 分布:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || echo "  (无 cpufreq —— 虚拟机中常见)"
echo "min/max: $(cat_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) / $(cat_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz"
echo "intel_pstate no_turbo: $(cat_or /sys/devices/system/cpu/intel_pstate/no_turbo)"
echo "--- C-state / idle ---"
echo "idle driver: $(cat_or /sys/devices/system/cpu/cpuidle/current_driver)"
echo "idle governor: $(cat_or /sys/devices/system/cpu/cpuidle/current_governor)"
for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
    [ -d "$s" ] && echo "  $(basename $s): $(cat_or $s/name) latency=$(cat_or $s/latency)us disable=$(cat_or $s/disable)"
done
echo "--- 拓扑（前 16 个逻辑核）---"
printf "%-5s %-8s %-8s %-8s %-10s\n" cpu core die cluster siblings
for c in $(seq 0 15); do
    d=/sys/devices/system/cpu/cpu$c/topology; [ -d "$d" ] || continue
    printf "%-5s %-8s %-8s %-8s %-10s\n" "$c" "$(cat_or $d/core_id)" "$(cat_or $d/die_id)" \
        "$(cat_or $d/cluster_id)" "$(cat_or $d/thread_siblings_list)"
done

sec NUMA
have numactl && numactl -H || echo "(无 numactl)"
have lscpu && lscpu | grep -i numa

sec GPU
have nvidia-smi || { echo "(无 nvidia-smi)"; }
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,uuid,serial,driver_version,vbios_version,persistence_mode,compute_mode,ecc.mode.current,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,temperature.gpu --format=csv
  echo "--- 虚拟化模式 / MIG ---"
  nvidia-smi -q 2>/dev/null | grep -iE "Virtualization Mode|Host VGPU|MIG Mode|Compute Mode|Addressing Mode" | head -12
  echo "--- 拓扑 ---"
  nvidia-smi topo -m 2>/dev/null | head -20
  echo "--- 占卡进程（应为空）---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
echo "--- PCIe 设备 ---"
for d in /sys/bus/pci/devices/*; do
  [ -r "$d/vendor" ] || continue
  [ "$(cat $d/vendor)" = "0x10de" ] || continue
  echo "  $(basename $d): numa_node=$(cat_or $d/numa_node) local_cpus=$(cat_or $d/local_cpulist) \
max_link=$(cat_or $d/max_link_speed)/$(cat_or $d/max_link_width) cur_link=$(cat_or $d/current_link_speed)/$(cat_or $d/current_link_width)"
done
have lspci && lspci -nn -d 10de: 2>/dev/null

sec CUDA
have nvcc && nvcc --version | tail -3 || echo "(无 nvcc)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>}"

echo "--- ★实际会被加载的 libcuda★（本项目头号变量，务必看这里）---"
# 不能只 ls /usr/lib，容器里 cuda-compat 包提供的 UMD 会通过 LD_LIBRARY_PATH 抢先被加载，
# 而它和宿主机注入的那个可能是完全不同的驱动分支（实测差 1.42x，见 README）。
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(未设置)}"
RESOLVED=""

# 探针一：bin/perf_bugs。它 NEEDED libcuda.so.1，是被测的那个真实负载，所以最可信。
# 公开检出里没有这个二进制（NVIDIA 内部工具），会落到探针二。
PB="$(dirname "$0")/bin/perf_bugs"
if [ -x "$PB" ]; then
    R=$(ldd "$PB" 2>/dev/null | sed -n 's/.*libcuda\.so\.1 => \(\S*\).*/\1/p')
    [ -n "$R" ] && { RESOLVED="$R"; echo "动态链接器解析到: $R   (探针: bin/perf_bugs)"; }
fi

# 探针二：让 python 自己 dlopen 一把，再从 /proc/self/maps 里读回实际映射的路径。
# 这里**不能**用 `ldd nvidia-smi` —— nvidia-smi 的 ELF 里根本没有 libcuda.so.1（连 NVML 都是
# 运行时 dlopen 的，readelf -d 只有 libc/libm/libdl/libpthread/librt），那个旧回退分支永远
# 匹配不到，是死代码；而 perf_bugs 总是在，于是一直没人发现。
# dlopen 走的是动态链接器的真实规则，LD_LIBRARY_PATH / ld.so.conf / RPATH 的覆盖都会如实反映
# 出来，这正是本项要回答的问题。只加载不调用 cuInit，不建上下文、不占显存。
if [ -z "$RESOLVED" ] && have python3; then
    R=$(python3 -I -c "
import ctypes, sys
try:
    ctypes.CDLL('libcuda.so.1')
except OSError:
    sys.exit(1)
for line in open('/proc/self/maps'):
    if 'libcuda.so' in line:
        print(line.split()[-1]); break
" 2>/dev/null)
    [ -n "$R" ] && { RESOLVED="$R"; echo "动态链接器解析到: $R   (探针: python dlopen)"; }
fi

if [ -n "$RESOLVED" ]; then
    ls -l "$RESOLVED"
    REAL=$(readlink -f "$RESOLVED"); echo "真实文件: $REAL  ($(stat -c%s "$REAL" 2>/dev/null) 字节)"
else
    echo "(两个探针都没能解析出 libcuda —— 可能没装驱动，或 python3 不可用；退回列目录)"
fi
echo "--- 系统中所有的 libcuda（可能不止一个！）---"
for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/cuda/compat/lib \
         /usr/local/cuda/compat/lib.real /usr/local/nvidia/lib64; do
    ls -l "$d"/libcuda.so.* 2>/dev/null
done | sort -u || echo "  (未找到)"

echo "--- 指令数采集的前置条件 ---"
echo "perf_event_paranoid: $(cat_or /proc/sys/kernel/perf_event_paranoid)   (<=3 或持有 CAP_PERFMON 才能读 PMC)"
echo "uid=$(id -u)"; grep -E '^Cap(Prm|Eff|Bnd)' /proc/self/status 2>/dev/null
grep -E '^Seccomp:' /proc/self/status 2>/dev/null
ls -d /usr/local/cuda* 2>/dev/null
have python3 && python3 -c "import torch;print('torch:',torch.__version__,'cuda:',torch.version.cuda)" 2>/dev/null || echo "torch: (不可用)"

sec IOMMU与内核参数
echo "cmdline: $(cat_or /proc/cmdline)"
echo "iommu_groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"

sec 系统负载
echo "loadavg: $(cat_or /proc/loadavg)"
have uptime && uptime
echo "内存: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB 总, "$7" GB 可用"}')"
echo
echo "========== 采集完成 =========="
