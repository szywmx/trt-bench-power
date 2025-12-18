# TensorRT-LLM Power Benchmark 使用指南

## 概述

这个工具基于 `llama-bench-power` 的设计理念，为 TensorRT-LLM 引擎提供类似的基准测试功能。它可以：

1. **测试不同的提示长度和生成令牌数量**
2. **监控功耗、温度和频率**（Jetson设备）
3. **记录详细的延时信息**
4. **输出JSON格式的结果用于分析**

## 快速开始

### 1. 准备环境

确保已安装 TensorRT-LLM：

```bash
python3 -c "import tensorrt_llm; print('OK')"
```

### 2. 准备TensorRT引擎

首先需要构建TensorRT引擎。参考 `tensorRT.md` 文档中的步骤：

```bash
# 转换检查点
python3 convert_checkpoint.py \
    --model_dir /path/to/model \
    --output_dir tllm_checkpoint \
    --dtype float16

# 构建引擎
trtllm-build \
    --checkpoint_dir tllm_checkpoint \
    --output_dir engine_dir \
    --gemm_plugin float16
```

### 3. 运行基准测试

#### 基本用法

```bash
python3 trt_bench_power.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer \
    -p 512 1024 \
    -n 128 256 \
    -r 10 \
    --use_mmap
```

#### 测试不同场景

**只测试提示处理（Prompt-only）:**
```bash
python3 trt_bench_power.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer \
    -p 128 256 512 1024 \
    -n 0 \
    -r 10
```

**只测试生成（Generation-only）:**
```bash
python3 trt_bench_power.py \
    --engine_dir TensorRT-LLM/examples/llama/engine_1gpu_gptq \
    --tokenizer_dir /mnt/sd0/llama/dir \
    -p 0 \
    -n 64 \
    -r 10
```

**测试提示+生成组合:**
```bash
python3 trt_bench_power.py \
    --engine_dir TensorRT-LLM/examples/llama/engine_1gpu_gptq \
    --tokenizer_dir /mnt/sd0/llama/dir \
    -pg 512,128 1024,256 \
    -r 10
```

## 参数说明

### 必需参数

- `--engine_dir`: TensorRT引擎目录路径
- `--tokenizer_dir`: 分词器目录路径

### 测试参数

- `-p, --n-prompt`: 要测试的提示长度列表（例如：`-p 128 256 512`）
- `-n, --n-gen`: 要测试的生成令牌数量列表（例如：`-n 64 128 256`）
- `-pg, --prompt-gen`: 提示和生成的组合对（例如：`-pg 512,128 1024,256`）
- `-r, --repetitions`: 每个测试的重复次数（默认：5）

### 输出参数

- `--output-dir`: 结果输出目录（默认：`./trt_bench_results`）
- `-o, --output`: 输出格式：json, jsonl, csv（默认：json）

### TensorRT-LLM特定参数

- `--use_mmap`: 使用内存映射（推荐，可降低内存占用）
- `--max_input_length`: 最大输入长度（默认：2048）
- `--log_level`: 日志级别：DEBUG, INFO, WARNING, ERROR（默认：WARNING）

## 使用测试脚本

`run_trt_bench_test.sh` 脚本可以自动化测试过程，类似于 `run_default_dvfs_test.sh`。

### 1. 编辑脚本配置

打开 `run_trt_bench_test.sh`，修改以下配置：

```bash
# 设置引擎和分词器路径
engine_configs=(
    "Llama-3.1-8B-Instruct-FP16:/path/to/engine1:/path/to/tokenizer1"
    "Llama-3.1-8B-Instruct-INT4:/path/to/engine2:/path/to/tokenizer2"
)

# 设置测试参数
n_repeat=10
n_p=0
n_d=1024
```

### 2. 运行脚本

```bash
chmod +x run_trt_bench_test.sh
./run_trt_bench_test.sh
```

脚本会：
- 重置CPU/GPU/EMC频率到默认DVFS模式
- 对每个引擎配置运行基准测试
- 记录系统状态和频率信息
- 在测试之间添加冷却时间
- 保存所有结果到输出目录

## 输出结果

### 文件结构

```
trt_bench_results/
├── trt_bench_summary_1234567890.json      # 汇总结果
├── trt_data_both_p512_g128_1234567890.json  # 详细数据（提示+生成）
├── trt_data_prompt_p1024_0_1234567891.json  # 详细数据（仅提示）
└── trt_data_generation_p0_g256_1234567892.json  # 详细数据（仅生成）
```

### 汇总JSON格式

```json
[
  {
    "test_type": "both",
    "n_prompt": 512,
    "n_gen": 128,
    "n_tokens": 640,
    "repetitions": 10,
    "avg_ns": 1234567890,
    "stddev_ns": 12345678,
    "avg_ts": 518.23,
    "stddev_ts": 5.12,
    "samples_ns": [1234567890, ...],
    "data_file": "./trt_bench_results/trt_data_both_p512_g128_1234567890.json",
    "engine_dir": "/path/to/engine",
    "tokenizer_dir": "/path/to/tokenizer"
  }
]
```

### 详细数据JSON格式

```json
{
  "power": [
    {
      "sampling_duration_ms": 5,
      "current": {
        "VDD_GPU_SOC": {
          "timestamps": [1234567890, ...],
          "value": [2.5, ...]
        },
        ...
      },
      "start_time_ns": 1234567890,
      "end_time_ns": 1234567899,
      "duration_ns": 9000000
    }
  ],
  "temp": [...],
  "freq": [...],
  "time": [
    {
      "n_p_eval": 512,
      "n_eval": 128,
      "rep_id": 0,
      "start_time_ns": 1234567890,
      "end_time_ns": 1234567899
    }
  ]
}
```

## 与 llama-bench-power 的对比

| 特性 | llama-bench-power | trt-bench-power |
|------|-------------------|-----------------|
| 框架 | llama.cpp | TensorRT-LLM |
| 模型格式 | GGUF | TensorRT Engine |
| 提示测试 | ✅ | ✅ |
| 生成测试 | ✅ | ✅ |
| 组合测试 | ✅ | ✅ |
| 功耗监控 | ✅ | ✅ |
| 温度监控 | ✅ | ✅ |
| 频率监控 | ✅ | ✅ |
| 时间记录 | ✅ | ✅ |
| 输出格式 | JSON/CSV/MD | JSON/JSONL/CSV |

## 注意事项

1. **提示处理测试**: 由于TensorRT-LLM的API特性，提示处理测试实际上会生成1个token。对于长提示，时间主要由提示处理主导。

2. **内存使用**: 使用 `--use_mmap` 标志可以显著降低内存使用，特别是在Jetson设备上。

3. **系统稳定性**: 在运行基准测试前，确保系统处于稳定状态。测试脚本会在测试之间添加冷却时间。

4. **监控限制**: 功耗、温度和频率监控仅在Jetson设备上可用。在其他平台上，工具会继续运行但不进行监控。

5. **引擎构建**: 确保TensorRT引擎是针对你的测试配置（最大输入/输出长度）构建的。

## 故障排除

### 导入错误

```
Error importing TensorRT-LLM: No module named 'tensorrt_llm'
```

**解决方案**: 确保TensorRT-LLM已正确安装：
```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

### 引擎加载错误

```
Error initializing ModelRunner: ...
```

**解决方案**: 
- 检查引擎目录路径是否正确
- 确保引擎是为正确的架构构建的
- 检查引擎文件是否完整

### 监控不工作

如果监控功能不工作，检查：
- 是否在Jetson设备上运行
- 监控文件路径是否存在（例如：`/sys/bus/i2c/drivers/ina3221/...`）
- 是否有读取权限

### 内存不足

如果遇到内存问题：
- 使用 `--use_mmap` 标志
- 减少最大输入/输出长度
- 确保有足够的swap空间

## 示例工作流

### 完整测试流程

1. **构建引擎**:
```bash
trtllm-build --checkpoint_dir checkpoint --output_dir engine --gemm_plugin float16
```

2. **运行基准测试**:
```bash
python3 trt_bench_power.py \
    --engine_dir /mnt/sd0/TensorRT-LLM/examples/llama/engine_1gpu_gptq \
    --tokenizer_dir /mnt/sd0/llama/dir \
    -p 0 \
    -n 1024 \
    -r 10 \
    --use_mmap \
    --output-dir /mnt/sd0/results
```

3. **分析结果**:
```bash
# 查看汇总
cat results/trt_bench_summary_*.json | jq

# 查看详细数据
cat results/trt_data_*.json | jq '.time'
```

## 参考

- TensorRT-LLM 文档: https://github.com/NVIDIA/TensorRT-LLM
- llama-bench-power: `/home/ruan/workspace/llama.cpp/examples/llama-bench/llama-bench-power.cpp`

sudo trtllm-build \
--checkpoint_dir /mnt/sd0/Llama-3.1-8B-Instruct-convert \
--gpt_attention_plugin float16 \
--gemm_plugin float16 \
--output_dir /mnt/sd0/TensorRT-LLM/examples/llama/engine_llama1