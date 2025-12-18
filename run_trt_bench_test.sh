#!/bin/bash

# TensorRT-LLM Power Benchmark Test Script
# Similar to run_default_dvfs_test.sh but for TensorRT-LLM

reset_cpu_freq() {
    # Reset CPU frequency range to default values, allowing DVFS dynamic adjustment
    local cpu_max_freq=2201600
    local cpu_min_freq=1036800
    for cpu_id in $(seq 0 $(($(nproc --all)-1))); do
        echo $cpu_max_freq > "/sys/devices/system/cpu/cpu${cpu_id}/cpufreq/scaling_max_freq"
        echo $cpu_min_freq > "/sys/devices/system/cpu/cpu${cpu_id}/cpufreq/scaling_min_freq"
    done
    local cur_cpu_max_freq=$(cat "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    local cur_cpu_min_freq=$(cat "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq")
    echo "Successfully reset CPU max and min frequency to $cur_cpu_max_freq and $cur_cpu_min_freq Hz"
}

reset_gpu_freq() {
    # Reset GPU frequency range to default values, allowing DVFS dynamic adjustment
    local gpu_max_freq=1300500000
    local gpu_min_freq=306000000
    echo $gpu_max_freq > "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq"
    echo $gpu_min_freq > "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq"
    local cur_gpu_max_freq=$(cat "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq")
    local cur_gpu_min_freq=$(cat "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq")
    echo "Successfully reset GPU max and min frequency to $cur_gpu_max_freq and $cur_gpu_min_freq Hz"
}

reset_emc_freq() {
    # Unlock EMC frequency, allowing DVFS dynamic adjustment
    echo 0 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
    echo 0 > /sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt
    echo "Successfully reset EMC frequency to default DVFS mode"
}

set_governor() {
    # Set CPU governor policy
    local governor=$1
    if [ -z "$governor" ]; then
        governor="performance"
    fi
    
    for cpu_id in $(seq 0 $(($(nproc --all)-1))); do
        local gov_path="/sys/devices/system/cpu/cpu${cpu_id}/cpufreq/scaling_governor"
        if [ -f "$gov_path" ]; then
            echo $governor > "$gov_path"
        fi
    done
    echo "Set CPU governor to $governor"
}

sleep_for_cool_down() {
    local cool_down_time=$1
    echo "Cooling down for $cool_down_time seconds..., current time: $(date +'%Y%m%d_%H%M%S')"
    sleep $cool_down_time
    echo "Cool down finished, current time: $(date +'%Y%m%d_%H%M%S')"
}

# ========== Configuration Parameters ==========
script_path="/home/ruan/workspace/trt-bench-power/trt_bench_power.py"

# Engine and tokenizer paths
# Update these paths according to your setup
engine_dir="/path/to/your/tensorrt/engine"
tokenizer_dir="/path/to/your/tokenizer"

# Model configurations
# You can test different engines by changing engine_dir
engine_configs=(
    "Llama-3.1-8B-Instruct-FP16:/path/to/engine1:/path/to/tokenizer1"
    "Llama-3.1-8B-Instruct-INT4:/path/to/engine2:/path/to/tokenizer2"
)

# Test parameters
n_repeat=10
n_p=0
n_d=1024
prompt_gen_param="-pg ${n_p},${n_d}"

# Output directory
output_dir="./trt_default_dvfs_test_rep${n_repeat}_p${n_p}_d${n_d}"
mkdir -p $output_dir

# ========== Initialize: Reset all frequencies to default DVFS mode ==========
echo "=========================================="
echo "Initializing: Resetting to default DVFS mode"
echo "=========================================="
reset_cpu_freq
reset_gpu_freq
reset_emc_freq
set_governor "performance"

echo ""
echo "Current system frequency status:"
echo "CPU0 governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "CPU0 min freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)"
echo "CPU0 max freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
echo "CPU0 cur freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
echo "GPU min freq: $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq)"
echo "GPU max freq: $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq)"
echo "GPU cur freq: $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq)"
echo "EMC rate locked: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked)"
echo "EMC cur rate: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/rate)"
echo ""

sleep_for_cool_down 30

# ========== Start Testing ==========
echo "=========================================="
echo "Starting TensorRT-LLM benchmark with default DVFS policy"
echo "=========================================="

for engine_config in "${engine_configs[@]}"
do
    IFS=':' read -r model_name engine_path tokenizer_path <<< "$engine_config"
    
    if [ ! -d "$engine_path" ]; then
        echo "Warning: Engine directory not found: $engine_path, skipping..."
        continue
    fi
    
    if [ ! -d "$tokenizer_path" ]; then
        echo "Warning: Tokenizer directory not found: $tokenizer_path, skipping..."
        continue
    fi
    
    current_datetime=$(date +'%Y%m%d_%H%M%S')
    
    echo "Running TensorRT-LLM benchmark with model ${model_name}, current time: ${current_datetime}"
    
    # Get current frequency information (for logging only)
    curr_cpu_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    curr_gpu_freq=$(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq)
    curr_emc_freq=$(cat /sys/kernel/debug/bpmp/debug/clk/emc/rate)
    
    log_dir="${output_dir}/trt_bench_${model_name}_default_dvfs"
    mkdir -p $log_dir
    log_file="${log_dir}/${current_datetime}.log"
    
    # Record test start information
    echo "=== Test Start Info ===" > ${log_file}
    echo "Model: ${model_name}" >> ${log_file}
    echo "Engine Dir: ${engine_path}" >> ${log_file}
    echo "Tokenizer Dir: ${tokenizer_path}" >> ${log_file}
    echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" >> ${log_file}
    echo "CPU Freq Range: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) - $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)" >> ${log_file}
    echo "GPU Freq Range: $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq) - $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq)" >> ${log_file}
    echo "EMC Locked: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked)" >> ${log_file}
    echo "Start CPU Freq: ${curr_cpu_freq}" >> ${log_file}
    echo "Start GPU Freq: ${curr_gpu_freq}" >> ${log_file}
    echo "Start EMC Freq: ${curr_emc_freq}" >> ${log_file}
    echo "=======================" >> ${log_file}
    echo "" >> ${log_file}
    
    # Run benchmark
    python3 $script_path \
        --engine_dir ${engine_path} \
        --tokenizer_dir ${tokenizer_path} \
        ${prompt_gen_param} \
        -r ${n_repeat} \
        --use_mmap \
        --output-dir ${log_dir} \
        -o json >> ${log_file} 2>&1
    
    # Record test end frequency information
    echo "" >> ${log_file}
    echo "=== Test End Info ===" >> ${log_file}
    echo "End CPU Freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)" >> ${log_file}
    echo "End GPU Freq: $(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq)" >> ${log_file}
    echo "End EMC Freq: $(cat /sys/kernel/debug/bpmp/debug/clk/emc/rate)" >> ${log_file}
    echo "=======================" >> ${log_file}
    
    sleep_for_cool_down 30
done

# ========== Test Complete ==========
echo "=========================================="
echo "TensorRT-LLM Benchmark completed"
echo "=========================================="
echo "Results saved in: $output_dir"
echo "System remains in default DVFS mode"
echo "done"

