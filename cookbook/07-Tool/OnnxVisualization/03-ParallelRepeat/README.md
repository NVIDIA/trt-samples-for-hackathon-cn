# Path B：图空间搜索（以及它到底该解决什么问题）

## 起因：DESIGN.md 的一个假设是错的

`03-PatternMiningFeasibility/DESIGN.md` 当时判断：1-D 降维路线**看不见并行重复**
（展开的 multi-head、MoE 的 N 个 expert），所以需要一条图空间的路径 B。

真去测的时候，手写的并行结构一个都没能复现这个假设：

| 结构 | path A 结果 |
| --- | --- |
| N 条独立分支 fan-out 后 `Sum` 汇合（MoE / multi-head 的形状） | ✅ 全部找到 |
| 分支中间读一个共享 hub | ✅ |
| 分支串在一条共享累加链上 | ✅ |
| torch 导出的 6-expert MoE | ✅ |

原因在理论上也说得通：

* 一个块能被 outline 的**前提**是它是**凸的**（收缩后不产生环），否则「算一半、出去、再回来」
  根本没法表达成一次函数调用；
* 一组两两不相交的凸节点集，收缩后仍是 DAG，因此**存在**某个拓扑序让它们全部连续；
* 所以 1-D 降维**并非天生看不见并行结构**，它唯一的弱点是在众多合法拓扑序里**挑错了一个**。

## 于是改成量化测量：`stress_test.py`

与其继续手工造反例，不如随机化：生成随机 DAG，在里面**植入 K 份同一个随机块**
（块的内部张量从不对外发布，所以按构造就是凸的），看工具能召回多少。

```bash
python3 stress_test.py 200
```

**只用 path A（`method="serial"`）**：

```
every instance recovered exactly :  112  (56.0%)
some exact, some partial         :   13
all instances only partial       :   69
planted block not touched at all :    6
numeric mismatch                 :    0
mean fraction of planted nodes recovered : 89.1%
```

失败样本长这样：

```
case  5 [partial] block=8n x5, recovered 100% of the planted nodes, got [(5,5), (3,5)]
case 21 [partial] block=7n x6, recovered 100% of the planted nodes, got [(4,6), (3,6)]
case  4 [partial] block=5n x2, recovered  80% of the planted nodes, got [(4,2)]
```

**真正的失败模式不是「并行重复看不见」，而是**：一个外来节点落在块的中间，
块在拓扑序里不再是连续的一段，于是它被**拆成两个模式**、或者**丢掉边界上的一两个节点**。
「块整个没找到」只占 6/200。

## 因此 path B 的重心也变了

原设计是「WL 小半径播种 → 实例组齐步生长」。实现后发现**播种这一半基本不起作用**：

> WL 颜色会把**块外的上下文**也编码进去。同一个块的两个实例通常挂在不同的张量上，
> 于是颜色不同、根本分不到一个桶。这正是 DESIGN.md 4.2.3 自己写过要小心的
> 「边界必须当通配符」，实现时还是踩了。

真正有效的是**齐步生长**这一半，但种子换成 **path A 已经对齐好的实例组**：

```
1-D 搜索给出一组对齐的实例（哪怕是截断的 / 拆开的）
        ↓
以它为种子，在图空间齐步生长：每一步所有实例沿同一条边扩张，否则谁也不扩
        ↓
P3 验证（凸性 / 对齐 / 接口），超界就逐个节点回退
        ↓
和 1-D 的原始候选按同一个 MDL 收益比大小，谁高用谁
```

关键点：

* **生长由「实例之间是否一致」驱动，不是由枚举驱动**，所以每一步只有一个后继状态，
  没有组合分支 —— 这是修正原思路「维护子模式集合不断加算子」的地方。
* **边的匹配键用节点的原始标签，绝不用 WL 颜色**，否则又会把块外上下文带进来。
* 两条路径的候选放在**同一个贪心循环里竞争**。分先后跑是没用的：
  1-D 会先把截断的块提交掉，图空间就没得改了（这是我第一版的写法，效果为零）。

`find_parallel_candidates`（WL 播种、从零生长）仍然保留，作为「1-D 完全没看到的块」的兜底，
但从实测看它贡献很小。

## 效果

`method="auto"`（默认）：

```
every instance recovered exactly :  173  (86.5%)     <- 56.0% 提升到 86.5%
some exact, some partial         :   11
all instances only partial       :   10              <- 69 降到 10
planted block not touched at all :    6
numeric mismatch                 :    0
mean fraction of planted nodes recovered : 94.6%     <- 89.1% 提升到 94.6%
```

真实模型（`00-Data/model/model-large.onnx`，1.5 GB / 6119 节点）：

| | `--method serial` | `--method auto` |
| --- | --- | --- |
| 主图节点 | 110 | **45** |
| function 数 | 3 | **1** |
| 最大模式 | 110 节点 × **23** 实例 | 110 节点 × **24** 实例 |
| 覆盖率 | 97.2% | **99.2%** |
| 耗时 | 46.1 s | **41.8 s** |
| ORT / TRT | pass / pass | pass / pass |

多找到了第 24 个实例，两个碎片模式（5×8、4×4）被吸收进主模式，而且**还快了一点**
（贪心迭代次数变少）。

6 层 TransformerEncoder 和 `02-ParallelTopology` 的分叉模型上两种 method 结果完全相同 ——
path A 本来就已经做对了，path B 不会把已经对的结果弄坏。

## 顺带修掉的一个缺陷

压力测试第一版报了 14 个「数值不一致」，`max_abs_diff = nan`。
不是改图错了，是模型本身会产生 NaN（`Sqrt` 负数、`Exp` 溢出），而 `nan != nan`
让两个逐字节相同的输出被判为不一致。校验改用 `np.array_equal(..., equal_nan=True)`。
修好后 200 个随机模型 **0 个数值错误**。

## 参数

```bash
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --method auto      # 默认：两条路径竞争
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --method serial    # 只用 1-D 降维
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --method parallel  # 只用图空间
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --wl-radius 1      # 从零生长时的锚点哈希半径（默认 1）
python3 -m tensorrt_cookbook.onnx_outliner model.onnx --beam 4           # 模式选择的束宽（默认 1 = 贪心）
```

## 第二轮：从 86.5% 顶到 96.0%

原以为瓶颈在「MDL 贪心不回头」，实测把失败按类型拆开后发现不是：

```
failure breakdown: exact 173, truncated_only 11, missing_instances_only 15, both 1
```

**主导的是「少了一个实例」（15 个），不是截断（11 个）。** 典型样本
`planted 6n x5 → got (6,4)`：块形状完全正确，就是漏了一个。

四步改进，每步都单独量化：

| 改进 | 精确召回 | 平均节点召回 |
| --- | ---: | ---: |
| 起点 | 86.5% | 94.6% |
| + **实例补全**（模式定下来后回图里找漏掉的实例） | 88.5% | 95.1% |
| + **边消歧**（生长时） | 92.0% | 95.9% |
| + **边消歧**（实例匹配时） | 93.0% | 96.1% |
| + **WL 半径 3 → 1** | **95.5%** | **98.6%** |
| + **beam=4** | 96.0% | 98.7% |

### 1. 实例补全（`find_more_instances`）

1-D 搜索的窗口必须是拓扑序里**连续的一段**，所以某个实例只要不连续就根本不会成为候选。
但模式一旦定下来，形状就已知了，回图里做一次结构匹配就能把它找回来 —— 很便宜。

### 2. 边消歧（贡献最大，+6.5pp）

原来的边匹配键是 `(方向, out_slot, in_slot, 邻居标签)`，要求**唯一匹配**。
但一个块经常把同一个值喂给两个**同类型**的节点：

```
edges of B0_0: [(out,0,0,'B0_1'), (out,0,0,'B0_2'), (out,0,1,'B0_4')]
                └──────── 同一个 key，两个不同的邻居 ────────┘
```

遇到歧义就跳过，结果是**三分之一的随机块把侧分支丢在了外面**。
改成用邻居**自身的边集签名**（`_local_signature`）来配对同 key 的多条边。
配错了也不会出错 —— `verify` 会挡掉，只是损失一个候选。

### 3. WL 半径 3 → 1（+2.5pp）

这条完全印证了 DESIGN.md 4.2.3 自己写过的警告，而我实现时把默认值设成了 3：

| `wl_radius` | 精确召回 | 平均节点召回 |
| ---: | ---: | ---: |
| 0 | 95.5% | 98.5% |
| **1（现默认）** | **95.5%** | **98.6%** |
| 2 | 93.5% | 96.6% |
| 3（原默认） | 93.0% | 96.1% |

半径越大，锚点哈希里就越多**块外的上下文**，而同一个块的两个实例挂在不同张量上，
于是它们的哈希不同、分不到一个桶。

### 4. beam search：实现了，但**不是杠杆**（+0.5pp）

`--beam N` 在「下一个提交哪个模式」上做束搜索（`beam=1` 就是原来的贪心）。

| beam | 精确召回 | transformer 耗时 | 大模型耗时 |
| ---: | ---: | ---: | ---: |
| 1（默认） | 95.5% | 2.9 s | 41.7 s |
| 2 | 95.5% | — | — |
| 4 | 96.0% | 2.9 s | **71.9 s** |

**+0.5pp 换 1.7 倍耗时**，所以默认仍是 1。真实模型上 1 和 4 结果完全相同。

> 实现时踩了一个坑：第一版 beam **越宽结果越差**（95.5% → 92.5% → 83.5%）。
> 这在逻辑上不可能——贪心的路径本来就在束里。查下来是**已经无法继续扩展的状态被丢出了束**，
> 于是一个更差但还能继续扩展的状态赢了。回归用例
> `beam_nested/beam_two_tower/beam_shared` 断言「束越宽总收益不会变低」，防止复发。

## 还剩的 4%

9 个失败里，有几个是**找到了收益相同的另一种切法**（比如植入 `3n x4` 收益 6，
找到 `4n x3` 收益也是 6），这不算真错 —— 植入的答案并非唯一最优。
其余是生长差一两个节点、或某个实例的上下文差异导致匹配失败。继续往上顶收益已经很小。
