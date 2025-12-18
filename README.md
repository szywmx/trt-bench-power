# TensorRT-LLM Power Benchmark Tool

This tool provides similar functionality to `llama-bench-power` but for TensorRT-LLM engines. It allows you to benchmark TensorRT-LLM models with different prompt lengths and generation token counts while monitoring power consumption, temperature, and frequency on Jetson devices.

## Features

- **Power Monitoring**: Monitor current consumption on Jetson devices (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0)
- **Temperature Monitoring**: Monitor CPU and GPU temperatures
- **Frequency Monitoring**: Monitor CPU, GPU, and EMC frequencies
- **Time Tracking**: Record detailed timing information for each inference call
- **Flexible Testing**: Support prompt-only, generation-only, and combined tests
- **Multiple Repetitions**: Run multiple repetitions for statistical accuracy
- **JSON Output**: Export results in JSON format for easy analysis

## Requirements

- Jetson device (AGX Orin, etc.) with TensorRT-LLM installed
- Python 3.8+
- TensorRT-LLM Python package
- PyTorch
- NumPy

## Installation

1. Ensure TensorRT-LLM is properly installed:
```bash
python3 -c "import tensorrt_llm; print('OK')"
```

2. Make scripts executable:
```bash
chmod +x trt_bench_power.py
chmod +x run_trt_bench_test.sh
```

## Usage

### Basic Usage

```bash
python3 trt_bench_power.py \
    --engine_dir /path/to/tensorrt/engine \
    --tokenizer_dir /path/to/tokenizer \
    -p 512 1024 \
    -n 128 256 \
    -r 10 \
    --use_mmap
```

### Parameters

- `--engine_dir`: Path to TensorRT-LLM engine directory (required)
- `--tokenizer_dir`: Path to tokenizer directory (required)
- `-p, --n-prompt`: Prompt lengths to test (default: [512])
- `-n, --n-gen`: Generation lengths to test (default: [128])
- `-pg, --prompt-gen`: Combined prompt,gen pairs (e.g., "512,128")
- `-r, --repetitions`: Number of repetitions per test (default: 5)
- `--output-dir`: Output directory for results (default: ./trt_bench_results)
- `-o, --output`: Output format: json, jsonl, or csv (default: json)
- `--use_mmap`: Use memory mapping (recommended for Jetson)
- `--max_input_length`: Maximum input length (default: 2048)
- `--log_level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: WARNING)

### Example: Testing Different Configurations

```bash
# Test prompt-only
python3 trt_bench_power.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer \
    -p 128 256 512 1024 \
    -n 0 \
    -r 10

# Test generation-only
python3 trt_bench_power.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer \
    -p 0 \
    -n 64 128 256 512 \
    -r 10

# Test combined prompt+generation
python3 trt_bench_power.py \
    --engine_dir /path/to/engine \
    --tokenizer_dir /path/to/tokenizer \
    -pg 512,128 1024,256 \
    -r 10
```

### Using the Test Script

The `run_trt_bench_test.sh` script automates the testing process similar to `run_default_dvfs_test.sh`:

1. Edit the script to set your engine and tokenizer paths:
```bash
engine_configs=(
    "Llama-3.1-8B-Instruct-FP16:/path/to/engine1:/path/to/tokenizer1"
    "Llama-3.1-8B-Instruct-INT4:/path/to/engine2:/path/to/tokenizer2"
)
```

2. Run the script:
```bash
./run_trt_bench_test.sh
```

## Output Format

The tool generates two types of output:

1. **Summary JSON**: Contains aggregated statistics for all tests
   - Average and standard deviation of tokens/second
   - Timing information
   - Test configuration

2. **Detailed Data JSON**: Contains detailed monitoring data for each test
   - Power consumption over time
   - Temperature over time
   - Frequency over time
   - Per-inference timing information

### Example Output Structure

```
trt_bench_results/
├── trt_bench_summary_1234567890.json
├── trt_data_both_p512_g128_1234567890.json
├── trt_data_prompt_p1024_0_1234567891.json
└── trt_data_generation_p0_g256_1234567892.json
```

## Comparison with llama-bench-power

| Feature | llama-bench-power | trt-bench-power |
|---------|------------------|-----------------|
| Framework | llama.cpp | TensorRT-LLM |
| Model Format | GGUF | TensorRT Engine |
| Monitoring | ✅ | ✅ |
| Prompt Testing | ✅ | ✅ |
| Generation Testing | ✅ | ✅ |
| Combined Testing | ✅ | ✅ |
| Output Format | JSON/CSV/Markdown | JSON/JSONL/CSV |

## Notes

- Power monitoring is only available on Jetson devices
- The tool uses TensorRT-LLM's Python API (`ModelRunner`)
- Memory mapping (`--use_mmap`) is recommended for Jetson devices to reduce memory usage
- For accurate results, ensure the system is in a stable state before running benchmarks
- Use the cool-down period between tests to allow the system to stabilize

## Troubleshooting

1. **Import Error**: Ensure TensorRT-LLM is properly installed and in your Python path
2. **Engine Loading Error**: Verify the engine directory path and that the engine was built correctly
3. **Monitoring Not Working**: Check that you're running on a Jetson device and that the monitoring paths exist
4. **Memory Issues**: Use `--use_mmap` flag and ensure sufficient swap space

## License

This tool is provided as-is for benchmarking purposes. Please refer to TensorRT-LLM's license for usage terms.

