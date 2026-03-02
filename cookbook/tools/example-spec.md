# unit_test.yaml 最小可落地规范（v1）

> 目标：
> - 保持示例目录可单独运行
> - 将测试编排集中到 `tools/run_examples.py`
> - 支持渐进迁移，先跑起来，再逐步替换旧 `unit_test.sh`

## 1) 文件位置

每个示例目录可选放置一个 `unit_test.yaml`：

- `cookbook/02-API/Layer/Cast/unit_test.yaml`
- `cookbook/05-Plugin/BasicExample/unit_test.yaml`

如果目录没有 `unit_test.yaml` 但有 `main.py`，runner 会使用默认行为：

- `run = ["python3 main.py > log-main.py.log"]`

## 2) 字段定义（v1）

```yaml
version: 1                # 可选，默认 1
name: Cast / simple       # 可选，展示名
enabled: true             # 可选，默认 true

# 过滤字段
tags: [api, layer, cast]  # 可选，字符串数组

# 执行控制
timeout: 1200             # 可选，单条命令超时（秒）
env:                       # 可选，注入环境变量
  MY_FLAG: "1"

# 生命周期命令（字符串或字符串数组）
pre:                       # 可选，run 前执行
  - "make test"
run:                       # 必填（在使用 unit_test.yaml 时）
  - "python3 main.py > log-main.py.log"
post:                      # 可选，run 后执行
  - "python3 verify.py"

# --clean 时追加执行
clean:                     # 可选
  - "rm -rf *.log"
```

约束：

- `run` 在存在 `unit_test.yaml` 时必须非空
- `pre/run/post/clean` 支持：
  - 单个字符串
  - 字符串数组
- `env` 的 value 支持标量（string/int/float/bool），会转换成字符串

## 3) 推荐迁移策略

1. 先引入 runner，不改示例代码
2. 对“特殊目录”补 `unit_test.yaml`（如需要 `make test`、`main.sh`）
3. 普通目录不写配置，直接走默认 `main.py` 规则
4. 最后把旧 `unit_test.sh` 改成薄壳或移除

## 4) 参数接口（run_examples.py）

基础：

- `--root PATH`：仓库根目录（默认自动定位到 `cookbook`）
- `--list`：只列出可运行示例
- `--dry-run`：只打印命令，不执行
- `--summary-json PATH`：输出 JSON 汇总报告

选择：

- `--case REL_PATH`：精确运行某个相对路径（可重复）
- `--include GLOB`：按 glob 包含（可重复，默认 `**`）
- `--exclude GLOB`：按 glob 排除（可重复）
- `--tags TAG`：只运行包含任一 tag 的示例（可重复）
- `--exclude-tags TAG`：排除带指定 tag 的示例（可重复）
- `--gpu-only`：只运行 `requires_gpu=true` 的示例

执行策略：

- `--timeout SEC`：默认每条命令超时（默认 1800）
- `--fail-fast`：首个失败即停止
- `--clean`：注入 `TRT_COOKBOOK_CLEAN=1`，并执行 `clean`

## 5) 配置样例

### 5.1 普通目录（可不写）

无 `unit_test.yaml`，目录内有 `main.py` 即可。

### 5.2 需要额外步骤的目录

```yaml
version: 1
tags: [plugin, compile]
pre:
  - "make test"
run:
  - "python3 main.py > log-main.py.log"
clean:
  - "make clean"
  - "rm -rf *.log"
```

### 5.3 非 main.py 入口目录

```yaml
version: 1
tags: [tool, polygraphy]
run:
  - "chmod +x main.sh"
  - "./main.sh"
  - "polygraphy run --help > Help-run.txt"
clean:
  - "rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt polygraphy_run.py"
```
