# `Loop` 折叠后端（M3）

前面所有折叠都是 **local function**，而 TensorRT 会把它内联回去 ——
对 build 时间和 engine **零影响**，纯粹服务可视化。

`--backend loop` 是**唯一真正能缩小 TensorRT 网络**的折叠方式：
ONNX `Loop` 会被映射成真正的 `ILoopLayer`，body 只实例化**一次**，与迭代次数无关。

```bash
python3 -m tensorrt_cookbook.onnx_outliner model.onnx -o model-loop.onnx --backend loop
```

## 代价与收益（12 层 TransformerEncoder，d_model=128，**B200**，2026-09-06 重测）

```
            ONNXnode  function  TRTlayer  build(s)  engine(MB)  latency(ms)   ORTdiff   TRTdiff
flat             492         0       858     16.61         9.6        0.412   0.0e+00   0.0e+00
function          12         1       858     12.81         9.6        0.471   0.0e+00   2.0e-06
loop               1         0       115      8.86         9.3        0.582   4.5e-08   2.0e-06
```

* **TRT layer 858 → 115（7.5x 更少）** —— 这是 `Loop` 唯一能给、function 给不了的东西。
* **build 16.6s → 8.9s**（H100 上是 12.2s → 8.1s；B200 上 flat 更慢，收益反而更大）。
* **engine 9.6MB → 9.3MB，基本不变**：权重量没变，只是换了存放形式。**折叠不省显存。**
* **延迟 0.412ms → 0.582ms（1.41x 慢）**：body 里的权重变成运行期 `Gather` 的结果，
  TRT 无法常量折叠、无法按权重挑专用 kernel、无法跨迭代融合。
* `function` 版与 `flat` **逐层相同**（858 vs 858），印证了 parser 会内联。

跑这个基准：

```bash
python3 main.py
```

> 基准脚本里有一处**必须重新随机化权重**：`nn.TransformerEncoder(layer, N)` 用
> `deepcopy` 克隆同一个层，开箱即用时 N 层权重**完全相同**，TensorRT 会在 engine 里
> 把它们去重（engine 只有 1.1MB 而不是 9.5MB），engine 体积的对比就失去意义了。
> 第一版基准就踩了这个坑，得到「Loop 让 engine 大 8 倍」的错误结论。

## `Loop` 的四种机制各对应什么

| ONNX `Loop` 机制 | 用来表达 |
| --- | --- |
| **loop-carried 变量** `v_1..v_N` | 迭代交给**下一个迭代**的每个值。**可以有多个** |
| **scan output** | 迭代交给**外部**的每个值。ONNX 把它们堆成 `[K,...]`，再用 `Gather(scan, k)` 把每片发回原消费者 |
| **读外层作用域** | 逐迭代相同的输入。子图**可以**读外层作用域，function 不行 |
| **`Gather(W_stacked, iter)`** | 逐迭代不同的 initializer，堆成一个 `[K,...]` 常量 |

一个值可以**同时**是 loop-carried 和 scan output（中间迭代既喂给下一迭代、又发布到外面），
此时 body 里要用 `Identity` 复制成两个不同的输出 —— ONNX 按**位置**而非名字识别 body 输出。

`analyse()` 会给出**精确的拒绝理由**，从不静默失败。剩余的硬限制：

* 实例必须构成**链**（一个实例最多喂给一个其他实例，且不能两个实例喂同一个）；
* 输出不能被**非相邻**的同链迭代消费；
* 输入必须是 loop-carried / 逐迭代相同 / 可堆叠 initializer 三者之一。

### 两个关键的放宽（都是实测逼出来的）

1. **多个 loop-carried 槽承载同一个值是允许的** —— transformer 的残差连接把块输入
   同时喂给 attention 和后面的 `Add`，那是**两个槽、一个循环变量**。
   第一版报 `2 loop-carried inputs, exactly one is required`，卡在这里。
2. **多个 loop-carried 变量是允许的** —— 一个块可能同时把隐状态和残差交给下一层。
   第一版只支持一个循环变量，`model-large.onnx` 因此被拒（`external output 0 is
   consumed by another iteration of the same chain`）。放宽后它需要 **2 个循环变量**。

### 多输出的代价：切片节点

scan output 要给**每个迭代的每个消费者**补一个 `Gather(scan, k)`。
`model-large.onnx` 上是 24 迭代 × 2 个 scan output = **48 个切片节点**，
所以 Loop 版的主图节点数（70）反而比 function 版（45）**多**。
但 TRT layer 数是 **5899 → 503**，切片节点本身极廉价。

**主图节点数变多、TRT 网络大幅变小**，这两件事不矛盾。

### 链拆分

一个模式不一定是一条链。6 层 encoder 排成 `(3 层 + tanh) × 2` 是**两条各 3 迭代的链**，
每条各折成一个 `Loop`（主图 8 → 4 节点，TRT 391 → 194 层）。

拆分是**全有或全无**的：只要有一个实例落单成长度 1 的链，整个模式退回 function，
而不是产出「一半 Loop 一半展开」的模型。

### 实测各模型的适用面

| 模型 | 结果 |
| --- | --- |
| 6 层 TransformerEncoder | ✅ 1 个 Loop，主图 **246 → 1** 节点，TRT 389 → **96** 层 |
| 12 层 TransformerEncoder | ✅ 1 个 Loop，TRT 858 → **115** 层 |
| 两级 `(3层+tanh)×2` | ✅ **2 个 Loop**（链拆分），TRT 391 → **194** 层 |
| 双塔（各 4 个块串联） | ✅ **2 个 Loop** |
| 4 个 MLP block 串联 | ✅ 1 个 Loop |
| 5 分支 fan-out | ❌ 退回 function：`the 5 instances break into 5 chains, 5 of them a single instance` |
| `02-ParallelTopology` 分叉模型 | ❌ 退回 function：同上（A 的输出被 B、C、Add 三处消费） |
| `00-Data/model-large.onnx`（1.5GB / 6119 节点） | ✅ 1 个 Loop：**24 迭代、2 个循环变量、2 个 scan output、48 个切片节点、12 个堆叠 initializer**，TRT **5899 → 503 层**（11.7x） |

## 数值：function 逐比特相同，Loop 不是（也不该要求它是）

| 后端 | ORT 相对误差 |
| --- | --- |
| `function` | **0.0（逐比特相同）** |
| `loop`（12 层 transformer） | **~3.6e-07**（约 3 个 fp32 ULP） |
| `loop`（`model-large.onnx`，24 迭代 49 输出） | **~4.5e-06**（逐输出看是 5e-07 ~ 2e-06，跨 4 组随机输入稳定） |

原因是权重不再是常量，onnxruntime 与 TensorRT 都会挑不同的 kernel，累加顺序变了。
这是舍入，不是逻辑错误 —— 逻辑错误会给出 O(1) 的偏差
（`02-ParallelTopology` 里抓到的那两个 bug 就是 6.0 和 1.298）。
误差随迭代深度累积，所以 24 层的模型比 12 层的大一个量级，这也是预期的。

校验的相对误差是**逐输出**计算的，不是「全局最大差 / 全局最大值」——
否则一个既有大输出又有小输出的模型会把小输出上的任意大误差藏起来。

所以校验容差**按后端取默认值**：`function` 用 0（要求逐比特），`loop` 用相对 1e-5。
`--tolerance` 可覆盖。报告里同时给出 `max_abs_diff`、`max_rel_diff` 和 `bit_exact`。

## 什么时候该用

**默认不要用。** 只在这两种情况下划算：

1. **build 时间难以忍受**而延迟不敏感（离线批处理、模型体积巨大导致 build 几十分钟）；
2. 需要**限制 TRT 网络规模**（层数过多触发 builder 的规模瓶颈）。

想要可读性 / 文件体积 / 上下游工具处理速度，用默认的 `--backend function`：
零风险、零性能损失、逐比特相同。
