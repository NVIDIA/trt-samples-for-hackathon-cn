# ChatGLM-6B TensorRT 流水线的候选项

来源：`/work/chatglm-6b`（856 行 Python 加一个 `Makefile`），一条面向客户的 ChatGLM-6B
PyTorch → ONNX → TensorRT 工作流，针对 **TensorRT 8.6** 编写，运行在 `nvcr.io/nvidia/pytorch:23.01-py3`
中。2026-08-28 阅读。**什么都没跑** —— 这台机器上没有 6B 的 checkpoint，而且代码针对的 TensorRT 版本
比 cookbook 落后三个大版本。

它早于 TensorRT-LLM 成为主流，用手工做了 TRT-LLM 后来产品化的那些事。这恰恰是它值得挖掘的理由：
**每一个技巧都摊在约 850 行里，而不是埋在一个框架里**，而且其中大多数根本不是模型特定的。

## 流水线的四个阶段

```txt
configuration.py  ──> configuration.yaml        # 一份生成的配置，被每个阶段消费
exportONNX.py     ──> onnx/TranAllInOne/model.onnx  (28 层 transformer，115 个 I/O 张量)
                      onnx/lm/model.onnx            (单独的 lm_head)
surgeon.py        ──> onnx/TranAllInOne/model-V1.onnx   # 4 个重写 pass，然后常量折叠
buildEngine.py    ──> plan/TranAllInOne.plan            # 约 12 GiB 的 FP16 engine
main.py           ──> prefill 一步 + 自回归 decode 循环
```

`Makefile` 用真实的文件级依赖把它们串起来，所以改动 `configuration.py` 只会让该失效的东西失效。
在 max-batch-size 1 下端到端约 25 分钟。

---

## 值得拿走的东西

| # | 技术 | 为什么能推广 | cookbook 目标 | 优先级 |
| - | ---- | ---- | ---- | ---- |
| C1 | **把整个 KV cache 融成一个 I/O 张量** | 28 层 × (K,V) = **56 输入 + 56 输出 → 1 + 1** | 03-Workflow（新建一个 LLM 形状的叶子）/ 04-Feature | **高** |
| C2 | **把 KV 输入和 KV 输出绑到同一个地址** | cache 原地增长；无拷贝、无乒乓 | 08-Advance 或 04-Feature | **高** |
| C3 | **把后处理搬进 engine** —— 切出最后一个位置 → MatMul(lm_head) → Softmax → ArgMax，于是 engine 返回一个 **token id** | 省掉**每生成一个 token** 一次 130528 宽的 D2H 拷贝 | 03-Workflow | **高** |
| C4 | **靠给模型自己的 `forward` 打补丁来导出**，而不是另写一个导出脚本 | 捕获的是你真正在跑的那张图，带着真实的 KV 输入 | 03-Workflow / 07-Tool | **高** |
| C5 | **`markGraphOutput` —— 一个图手术的调试原语** | 在任意节点处截断 ONNX，用来二分一个精度 bug | 07-Tool/OnnxGraphSurgeon | **高** |
| C6 | **在手术阶段把 rotary 的 cos/sin 预计算成常量表** | 普适教训：任何在整次运行中恒定的东西都不该留在图里 | 07-Tool/OnnxGraphSurgeon | 中高 |
| C7 | **就地改 `Transpose.perm` 来重写 attention 的布局** | 怎么把融合 MHA kernel 想要的布局递给它 | 07-Tool/OnnxGraphSurgeon | 中 |
| C8 | **把第二个 ONNX 文件作为 Constant 并入**（`lm_head` 的权重从它自己的 ONNX 里提出来） | 把两张导出的图合成一个 engine | 07-Tool/OnnxGraphSurgeon | 中 |
| C9 | **每个 `gs.Constant` 都要 `np.ascontiguousarray`** | 不这么做 TensorRT 会把这个 Constant 的形状读成 `(0)` —— 静默，作者留了一条吼叫式的注释 | 07-Tool/OnnxGraphSurgeon（陷阱注记） | 中 |
| C10 | **prefill 与 decode 是同一个 engine 上的两种形状状态** | 流水线先设 `[B,L]` 再设 `[B,1]`，`L_past` 从 0 到 N；这是多 optimization profile 的天然动机 | 关联 §1 的 P5 / `08-Advance/MultiOptimizationProfile` | 中 |
| C11 | **「生成配置 + Makefile」的流水线结构** | cookbook 的每个例子都是一个文件；四阶段工作流需要分段和缓存 | 03-Workflow（新叶子的结构） | 中低 |
| C12 | **在打过补丁的 PyTorch 模型里埋 NVTX range**（`TRAN` 绿色、`LM` 蓝色） | 在同一条 Nsight 时间线上把 torch 阶段和 TRT 阶段对齐 | 07-Tool/nvtx（交叉引用） | 低 |

### C1 —— 一个张量取代 112 个

`surgeon.py::adjustInputOutput` 把 56 个 KV 输入换成一个
`[(56*L_past), B, 32, 128]` 的张量加一个 `num_outputs=56` 的 `Split` 节点，并把 56 个 KV 输出
concat 成一个 `[56*(L_past+L), B, 32, 128]` 的张量：

```python
tensorInputKV = gs.Variable(..., sDataType, ['(56*L_past)', 'B', 32, 128])
nodeSplit = gs.Node("Split", ..., inputs=[tensorInputKV], outputs=tensorPadKVList,
                    attrs=OrderedDict([('num_outputs', 56)]))
...
graph.inputs  = graph.inputs[:3] + [tensorInputKV]
graph.outputs = [graph.outputs[0], tensorConcat]
```

115 个 I/O 张量变成 6 个。每一步因此少了 112 次 `set_tensor_address` / `set_input_shape` 调用，而
`main.py` 里的形状记账也坍缩成一行。这个想法不是 LLM 特有的：**任何带有一大把同形状 I/O 的模型，都
应该给它一个打包好的张量加一对 Split/Concat。**

### C2 —— cache 原地增长

```python
context.set_tensor_address(lTensorName[3], bufferKVCacheList.data_ptr())  # 输入
context.set_tensor_address(lTensorName[4], bufferKVCacheList.data_ptr())  # 输出，同一个指针
```

因为输出 cache 就是输入 cache 沿第 0 维再追加 `L` 条，所以输入和输出可以共用一块分配。
`backupFile/main-twoBuffer.py` 是更早的乒乓版本（`bufferKVCacheList = [.., ..]` 配
`iSource`/`iTarget`）—— 把两份都留着构成一组很好的前后对照：最终版本把 cache 内存减半并去掉了交换。
值得与 `05-Plugin/AliasedIOPlugin` 配对，那是低一层的同一个想法。

### C3 —— engine 返回的是一个 token，不是一个 logit 向量

`surgeon.py::addTail` 在 ONNX 里追加：

```txt
Shape → Gather → Unsqueeze → Sub      # 算出最后一个有效位置
Slice(最后一个位置)                     # [1, B, 4096]
MatMul(constantLM)                    # [1, B, 130528]   lm_head 作为 Constant 折进来
Softmax → ArgMax → Transpose          # [B, 1]           下一个 token id
```

`bKeepLogit` 可以把这个尾巴切回原始 logits，供调用方自己做采样。词表是 130528，所以贪心路径
**每步搬 2 个数而不是 261056 个**。可迁移的教训比采样本身更大：*如果主机对每个输出都做同一件固定的
事，就考虑把它放进图里。*

### C4 —— 靠给 `forward` 打补丁来导出

`exportONNX.py` 只有 14 行，而且**不含任何 export 调用**：

```python
model = transformers.AutoModel.from_pretrained("pyTorchModel", trust_remote_code=True).eval()
model.chat(tokenizer, query=..., history=[])   # "we will not wait for the response"
```

真正的导出住在模型自己 `modeling_chatglm.py` 的一份打了补丁的拷贝里
（`backupFile/modeling_chatglm.py`，由 `Makefile` 换进去），它在 `forward` **内部**调用
`torch.onnx.export` —— 并被守卫成**在第二次调用、即 `past_key_values` 不为 None 时**才触发，也就是说
它捕获的是带真实 cache 输入的 *decode* 图而不是一张合成的图，然后一旦两个文件都生成就退出。

这就是「我无法控制一个模型的 `forward` 签名，怎么在它真正被使用的状态下导出它」的答案。它也让那个
60 输入的 `dynamic_axes` 字典变成了一个推导式：

```python
dyDict = {"input_ids": {0: 'B', 1: 'L'}, "position_ids": {0: 'B', 2: 'L'}, "attention_mask": {0: 'B', 2: 'L', 3: 'L'}}
dyDict.update({"input_past_key_%d" % i: {0: 'L_past', 1: 'B'} for i in range(28)})
```

同一个补丁还带着两个有用的习惯：注释掉的代码块可以把每个输入/输出 dump 到 `io.npz` 用于精度调试，
以及 C12 的那些 NVTX range。

### C5 —— `markGraphOutput`

一个 30 行的 ONNX-GraphSurgeon 辅助函数，把任意节点的输出（或输入，或指定序号的那一个）标记为图输出，
并且默认**删除原来的输出**，从而截断整张图：

```python
markGraphOutput(graph, ["/Conv"])                     # 该节点的输出张量
markGraphOutput(graph, ["/Conv"], False, True)        # 它的输入（data + weight + bias）
markGraphOutput(graph, ["/TopK"], lMarkOutput=[1])    # 它的第二个输出
```

这是「FP16 engine 在 400 个节点里的某处不对」背后的实用工具 —— 在某个节点处剪断图、构建、比较、二分。
`07-Tool/OnnxGraphSurgeon/08_isolate_subgraph.py` 能隔离子图，但没有提供这套「二分精度 bug」的工作流；
它和 Polygraphy 的 `debug reduce` 材料（§3.3 的 L1）天然配对。

### C6 / C7 —— 两个模型形状的重写

`adjustPositionEmbedding` 在主机侧预计算 rotary 表，然后从表里 Gather：

```python
inv_freq   = 10 ** (-1 / 16 * np.arange(0, 64, 2, dtype=np.float32))
valueTable = (np.arange(nMaxSLPast) ⊗ concat(inv_freq, inv_freq)).reshape(nMaxSLPast, 1, 128)
constantCosTable, constantSinTable = gs.Constant(np.cos(valueTable)), gs.Constant(np.sin(valueTable))
```

`editAttention` 则找到每个 `Softmax` 并向外走（`node.i().i().i()`、
`nodeWhere.i(2).i().i().i().i()`），把周围的 Transpose 改成目标 `[B, 32, L, 128]` /
`[B, 32, 128, L]`，并在 `Where` 之前丢掉一个无用的 `*1+0`。

拿走**技术**（主机侧常量表；改 `perm` 并重新接线而不是插入新节点；以 `Softmax` 这样的地标算子作为
搜索锚点），不要拿代码 —— 那些 `.i().i().i()` 链条脆弱得和它们看起来一模一样，而且绑死在某一个模型的
某一次导出上。

### C9 —— 那个陷阱，用作者自己的话说

```python
# Constant of 1 dimension integer value, MUST use np.ascontiguousarray,
# or TRT will regard the shape of this Constant as (0) !!!
constant0 = gs.Constant("constant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
```

值得在 `07-Tool/OnnxGraphSurgeon` 里写一行注记 —— 一个非连续的数组会静默变成一个 rank-0 的常量。

---

## 不要拿走的东西

+ **那套 TensorRT 8.6 的构建配方在 TensorRT 11 下大半是非法的。** `buildEngine.py` 用了
  `create_network(1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH)`（该标志在 TRT 10 被移除）、
  `BuilderFlag.FP16` + `OBEY_PRECISION_CONSTRAINTS` 以及逐层的 `set_output_type`
  （**在 TRT 11 全部移除 —— 网络是强类型的**），还有
  `PreviewFeature.FASTER_DYNAMIC_SHAPES_0805`（已消失）。如果这要变成一个 cookbook 例子，构建阶段是
  重写而不是移植 —— 但这也让它成为 `trt-strong-typing-migration` 那个故事的好素材。
+ `config.builder_optimization_level = 5`，注释说是「为了用原生 TensorRT kernel 而不是 myelin，性能
  更差但支持多 batch」—— 一个时代特定的绕行方案，而且 README 自己也承认「这会在未来版本中修复」。
  不要在没有重测的情况下把这个说法带下去。
+ `profileList = [builder.create_optimization_profile() for i in range(2)]`，而实际上只有
  `profileList[0]` 被加进去过 —— 死代码，但它显示了作者当时的方向。见 C10。
+ 一切 ChatGLM 特有的东西：`mask_token_id`/`gmask_token_id`/`bos_token_id` 常量、两行的
  `position_ids`（position + block position）、`process_response` 里的中日韩标点正则、
  `[Round n]\n问：…\n答：` 的 prompt 格式、那些 `.i(2).i().i().i().i()` 的走法。
+ 12 GiB 的 engine / 15 GiB 的峰值内存和那个 6B checkpoint：任何 cookbook 版本都必须在一个很小的模型
  上重新表达，否则在 CI 里跑不起来。

## 已覆盖 —— 不要再提

timing cache 复用（`04-Feature/TimingCache`）、`hardware_compatibility_level = AMPERE_PLUS`
（`04-Feature/HardwareCompatibility`）、`builder_optimization_level`
（`04-Feature/BuilderOptimizationLevel`）、NVTX range（`07-Tool/nvtx`）、用 Polygraphy 做常量折叠
（`07-Tool/OnnxGraphSurgeon/06_fold.py`）、基础的 PyTorch→ONNX→TRT
（`03-Workflow/pyTorch-ONNX-TensorRT`）。

## 建议的吸纳形态

不要做成一个大例子。分成两块：

1. **`07-Tool/OnnxGraphSurgeon` 获得这套工具箱** —— C5（`markGraphOutput` 二分）、C6（主机侧常量表）、
   C8（把第二个 ONNX 折成 Constant）、C9（连续性陷阱）。全都很小、自足，而且可以在那个目录里已有的
   模型上测试。
2. **新建一个 `03-Workflow` 叶子：「一个 decoder-only 模型，端到端」** —— 一个*玩具级*的 2 层
   transformer 带 KV cache，承载 C1（打包的 cache I/O）、C2（原地 cache，并把双缓冲版本作为「之前」）、
   C3（在 engine 里采样）、C4（从真实的 forward 导出）、C10（prefill/decode profile），模型小到几秒就能
   构建完。**这正是 cookbook 真正缺的那一块：它完全没有自回归的工作流。**

6B 模型本身留在原地 —— 把 `/work/chatglm-6b` 作为全尺寸参考引用即可。

---

## 在 gpt2-medium 上的可行性 —— **2026-08-28 验证，四条头部技术全部可迁移**

6B 模型对 cookbook 太大了，所以问题是 `gpt2-medium`（树里已有的
`00-Data/model/model-large.onnx`）能否承载同样的教训。下面的一切都是**跑出来的**，不是推理出来的；
导出和验证是在 `gpt2`（124M，同一架构家族，迭代快 10 倍）上做的，形状则与真实的 gpt2-medium ONNX
交叉核对过。

### KV cache 在那里，而且 I/O 爆炸比 ChatGLM 还严重

| 产物 | I/O 张量数 |
| ---- | ---- |
| `00-Data/model/model-large.onnx` = gpt2-medium 的 `decoder_model.onnx` | 2 入（`input_ids`、`attention_mask`）+ **49 出**（`logits` + 24 层 × `present.{i}.{key,value}`）—— **只有 prefill，没有 past 输入** |
| gpt2-medium 的 `decoder_with_past_model.onnx`（同一 HF 目录，1.63 GB，已下载并检查） | **50 入 + 49 出 = 99 个张量**，`past_key_values.{i}.{key,value}` 形状 `[B, 16, past, 64]` |
| 下面那个 wrapper | **3 入 + 2 出** |

所以 C1 适用，而且比在 ChatGLM 上更狠（99 个张量对 115 个）。gpt2-medium 是 24 层 × 16 头 × 64 头维；
同目录还有 `decoder_model_merged.onnx`（prefill+decode 藏在一个 `use_cache_branch` 的 `If` 后面），
这是值得知道的第三种产物。

### C4：gpt2 **不**需要这个 hack —— 而且 ChatGLM 的 hack 现在到哪儿都不管用了

ChatGLM 给 `modeling_chatglm.py` 打补丁，是因为它是一个 `trust_remote_code` 模型，你控制不了它的
`forward`，而且它的调用埋在 `model.chat()` 里面。`GPT2LMHeadModel` 是 `transformers` 的一等公民，
签名是公开的，所以这些都不成立。

更重要的是，**不管什么模型，ChatGLM 那套配方在现代 `transformers` 上都已经死了**。在 5.16.1 上
`past_key_values` 的类型是 `transformers.cache_utils.Cache`，而 ChatGLM 递给 `torch.onnx.export` 的
那种元组套元组会被直接拒绝：

```txt
legacy tuple REJECTED: AttributeError 'list' object has no attribute 'get_seq_length'
```

现代的等价物是一个约 15 行的 wrapper `nn.Module`，而且它**更好**：不用换文件、不用 Makefile 跳舞、
不用「第二次调用才触发」的把戏，而且 C1 的打包发生在 PyTorch 里而不是 ONNX-GraphSurgeon 里：

```python
class Gpt2Step(torch.nn.Module):
    def forward(self, input_ids, attention_mask, past_kv):   # past_kv: [LPast, 2N, B, H, D]
        p = past_kv.permute(1, 2, 3, 0, 4)
        cache = DynamicCache()
        for i in range(self.n_layer):
            cache.update(p[2 * i], p[2 * i + 1], i)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=cache, use_cache=True)
        c = out.past_key_values
        present = torch.stack([t for i in range(self.n_layer)
                               for t in (c.layers[i].keys, c.layers[i].values)], 0)
        present = present.permute(3, 0, 1, 2, 4).contiguous()          # [LTotal, 2N, B, H, D]
        next_token = torch.argmax(out.logits[:, -1, :], -1, keepdim=True).to(torch.int32)  # C3
        return next_token, present
```

`torch.onnx.export(..., opset_version=17, dynamo=False)` 成功。它会吐一堆 `TracerWarning`
（`if not self.is_initialized or self.keys.numel() == 0`、各种 mask 长度检查、
`is_causal = q_length > 1 ...`），看起来像是这张图被特化到了被 trace 的那个长度上 —— **并没有**，
下面验证了。`is_causal` **确实**被烤死了，这正是 prefill 和 decode 需要分开导出的原因；ChatGLM 当年
也是这么做的。

### 在非 trace 长度上与 PyTorch 对照验证

在 `LPast = 3` 处 trace，然后过 onnxruntime 并与 eager PyTorch 比较：

```txt
Lp= 1  onnx_token=  198 torch_token=  198 match=True  present (24,1,12, 2,64) maxdiff=3.62e-05
Lp= 3  onnx_token=  198 torch_token=  198 match=True  present (24,1,12, 4,64) maxdiff=1.81e-05
Lp= 7  onnx_token=   11 torch_token=   11 match=True  present (24,1,12, 8,64) maxdiff=2.29e-05
Lp=16  onnx_token=   68 torch_token=   68 match=True  present (24,1,12,17,64) maxdiff=2.48e-05
```

token id 完全一致；cache 只差 FP32 的噪声。

### C2 **不能**白拿 —— 而这恰恰是有意思的地方

ChatGLM 的 cache 是 `[L, B, 32, 128]`：序列轴在**最外层**，所以输出 cache 就是输入 cache 在分配的末尾
追加若干行，一块缓冲区就能同时当输入和输出。HuggingFace GPT-2 的 cache 是 `[B, H, L, D]` —— 序列轴
在**第 2 维**，于是新条目是交错插入的，共用缓冲区直接就是错的。

修法是按序列优先打包成 `[LPast, 2N, B, H, D]`，代价是图的两端各一次 `permute`。已验证 —— 输出的前
`LPast` 行与输入逐字节相同，这正是可以别名共用的前提：

```txt
Lp= 1 token=  198 present (2, 24,1,12,64) prefix-preserved=True
Lp= 5 token=  198 present (6, 24,1,12,64) prefix-preserved=True
Lp=12 token=   68 present (13,24,1,12,64) prefix-preserved=True
```

**这比原来那个教训更好。** ChatGLM 的原地 cache 看起来像个聪明的技巧；和 GPT-2 并排一看，它被揭示为
*cache 布局的一个后果*，而这个例子可以把「允许它的布局」和「不允许它的布局」都展示出来。

### 结论

在 gpt2-medium 上重写。C1 / C3 / C4 直接迁移而且变得更简单；C2 只有在序列优先的重打包之后才能迁移，
值得单独一个 case。**失去的是 C5-C9** 那批 ONNX-GraphSurgeon 材料 —— 因为那个 wrapper 在 PyTorch 里
做了 ChatGLM 不得不用图手术做的事。那些留作一次独立的 `07-Tool/OnnxGraphSurgeon` 吸纳（S6a），
源头仍是 ChatGLM 的代码；于是 S6 的两半变得真正互相独立。

`transformers` 是 cookbook 的一个新依赖（用 `pip install --dry-run` 查过：它只会拉进 `typer`、
`shellingham`、`annotated-doc` —— **不会碰 numpy 和 tensorrt**）。

### gpt2-small 还是 gpt2-medium —— 用 **small**

|  | gpt2 (small) | gpt2-medium |
| --- | --- | --- |
| 层 / 头 / 隐层 | 12 / 12 / 768 | 24 / 16 / 1024 |
| 参数量 | 124 M | 355 M |
| **词表** | **50257** | **50257** —— 完全相同 |
| `decoder_with_past_model.onnx` 的 I/O | 26 入 + 25 出 = **51** | 50 入 + 49 出 = **99** |
| 同一张图过 wrapper 之后 | **5** | **5** |
| HF 的 ONNX 下载量 | 623 MiB | 1552 MiB |
| 导出的单步图（实测 / 估计） | **652 MB** | 约 1.4 GB |
| TensorRT engine，FP32（实测 / 估计） | **476 MiB，11.7 秒构建完** | 约 1.2 GiB，约 30 秒 |

这个例子要教的一切都没有变：同样的架构、同样的 cache 布局、同样的导出路径，而且 —— 因为词表
**完全相同** —— C3 的那句论证（engine 每步返回一个 token id 而不是 50257 个浮点数）在数值上是同一句话。
唯一失去的是那句头条从「99 个 I/O 张量 → 5 个」变成「51 → 5」。

这个例子可以两者兼得：**case 1 打开 `00-Data` 里已经躺着的 `model-large.onnx`**（gpt2-medium，
2 入 + 49 出）来展示中等规模下的爆炸、零下载，然后可运行的那一半用 small。

### 在 small 上端到端 —— 已经验证

不只是导出：解析了、构建了、跑起来了，而且 `past_kv` 和 `present_kv` 绑在了**同一块分配**上。

```txt
engine I/O: ['input_ids', 'attention_mask', 'past_kv', 'next_token', 'present_kv']
prompt   : TensorRT is a
continued:  new type of neural network that can be used to train a
```

所以整个 S6b 例子的风险已经排除；剩下的只是按 cookbook 的风格把它包装起来。
