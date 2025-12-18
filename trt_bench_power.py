#!/usr/bin/env python3
"""
TensorRT-LLM Power Benchmark Tool
Similar to llama-bench-power but for TensorRT-LLM engines
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch

# Add TensorRT-LLM examples to path
sys.path.insert(0, str(Path(__file__).parent.parent / "TensorRT-LLM" / "examples"))

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner
    from tensorrt_llm.logger import logger
    from utils import load_tokenizer, DEFAULT_PROMPT_TEMPLATES
except ImportError as e:
    print(f"Error importing TensorRT-LLM: {e}")
    print("Please ensure TensorRT-LLM is properly installed")
    sys.exit(1)

from monitor_py import (
    is_jetson, JetsonPowerMonitor, JetsonTempMonitor, JetsonFreqMonitor,
    TimeInfoHolder, TimeInfoItem, dump_data_to_json
)


def get_current_time_ns() -> int:
    """Get current time in nanoseconds"""
    return int(time.time_ns())


def generate_random_tokens(tokenizer, n_tokens: int, pad_id: int) -> torch.Tensor:
    """Generate random token IDs for testing"""
    vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
    tokens = torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int32)
    return tokens


def test_prompt(
    runner: ModelRunner,
    tokenizer,
    n_prompt: int,
    pad_id: int,
    end_id: int,
    time_holder: TimeInfoHolder,
    rep_id: int
):
    """Test prompt processing (context phase)
    Note: TensorRT-LLM processes prompt during generation, so we generate 1 token
    to measure the context processing time. The time includes both prompt processing
    and first token generation, but is dominated by prompt processing for long prompts.
    """
    tokens = generate_random_tokens(tokenizer, n_prompt, pad_id)
    input_ids = tokens.unsqueeze(0)  # Add batch dimension [1, n_prompt]
    
    start_time = get_current_time_ns()
    # Generate 1 token to measure prompt processing + first token generation
    # For long prompts, this is dominated by prompt processing
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=1,  # Generate 1 token to complete the operation
            end_id=end_id,
            pad_id=pad_id,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            return_dict=True,
        )
    torch.cuda.synchronize()
    end_time = get_current_time_ns()
    
    time_holder.add_info(TimeInfoItem(
        n_p_eval=n_prompt,
        n_eval=1,  # We generate 1 token, but this is mainly for prompt processing
        rep_id=rep_id,
        start_time_ns=start_time,
        end_time_ns=end_time
    ))


def test_generation(
    runner: ModelRunner,
    tokenizer,
    n_gen: int,
    n_past: int,
    pad_id: int,
    end_id: int,
    time_holder: TimeInfoHolder,
    rep_id: int
):
    """Test token generation (decoding phase only)"""
    # Generate a random prompt token to start with
    start_token = generate_random_tokens(tokenizer, 1, pad_id)[0].item()
    input_ids = torch.tensor([[start_token]], dtype=torch.int32)
    
    start_time = get_current_time_ns()
    
    # Generate tokens
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=n_gen,
            end_id=end_id,
            pad_id=pad_id,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            return_dict=True,
        )
    torch.cuda.synchronize()
    end_time = get_current_time_ns()
    
    # Record time for the entire generation
    time_holder.add_info(TimeInfoItem(
        n_p_eval=n_past,
        n_eval=n_gen,
        rep_id=rep_id,
        start_time_ns=start_time,
        end_time_ns=end_time
    ))


def test_prompt_and_generation(
    runner: ModelRunner,
    tokenizer,
    n_prompt: int,
    n_gen: int,
    pad_id: int,
    end_id: int,
    time_holder: TimeInfoHolder,
    rep_id: int
):
    """Test both prompt processing and generation"""
    # Generate random prompt tokens
    prompt_tokens = generate_random_tokens(tokenizer, n_prompt, pad_id)
    input_ids = prompt_tokens.unsqueeze(0)  # Add batch dimension [1, n_prompt]
    
    start_time = get_current_time_ns()
    
    # Generate tokens
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=n_gen,
            end_id=end_id,
            pad_id=pad_id,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            return_dict=True,
        )
    torch.cuda.synchronize()
    end_time = get_current_time_ns()
    
    # Record time for the entire operation
    time_holder.add_info(TimeInfoItem(
        n_p_eval=n_prompt,
        n_eval=n_gen,
        rep_id=rep_id,
        start_time_ns=start_time,
        end_time_ns=end_time
    ))


def parse_arguments():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Power Benchmark Tool")
    
    # Model and engine
    parser.add_argument("--engine_dir", type=str, required=True,
                       help="Path to TensorRT-LLM engine directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                       help="Path to tokenizer directory")
    
    # Test parameters
    parser.add_argument("-p", "--n-prompt", type=int, nargs="+", default=[512],
                       help="Prompt lengths to test")
    parser.add_argument("-n", "--n-gen", type=int, nargs="+", default=[128],
                       help="Generation lengths to test")
    parser.add_argument("-pg", "--prompt-gen", type=str, nargs="+",
                       help="Prompt,gen pairs (e.g., '512,128')")
    parser.add_argument("-r", "--repetitions", type=int, default=5,
                       help="Number of repetitions per test")
    
    # Output
    parser.add_argument("-o", "--output", type=str, default="json",
                       choices=["json", "jsonl", "csv"],
                       help="Output format")
    parser.add_argument("--output-dir", type=str, default="./trt_bench_results",
                       help="Output directory for results")
    
    # TensorRT-LLM specific
    parser.add_argument("--use_mmap", action="store_true",
                       help="Use memory mapping")
    parser.add_argument("--max_input_length", type=int, default=2048,
                       help="Maximum input length")
    # Logging level (accept WARN/WARNING and map internally)
    parser.add_argument("--log_level", type=str, default="WARNING",
                       choices=["DEBUG", "INFO", "WARN", "ERROR", "WARNING",
                                "debug", "info", "warn", "error", "warning"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Set log level: TensorRT-LLM expects lowercase keys (warning/info/...)
    log_level = args.log_level.lower()
    if log_level == "warn":
        log_level = "warning"
    logger.set_level(log_level)
    
    # Check if running on Jetson
    if not is_jetson():
        print("Warning: Power monitoring is only supported on Jetson devices")
        print("Continuing without power monitoring...")
        use_monitoring = False
    else:
        use_monitoring = True
    
    # Load tokenizer
    try:
        tokenizer, pad_id, end_id = load_tokenizer(
            tokenizer_dir=args.tokenizer_dir,
            model_name="LlamaForCausalLM",
            model_version=None,
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Initialize ModelRunner
    try:
        max_output_len = max(args.n_gen) if args.n_gen and max(args.n_gen) > 0 else 512
        max_input_len = max(args.n_prompt) if args.n_prompt and max(args.n_prompt) > 0 else args.max_input_length
        
        runner_kwargs = {
            "engine_dir": args.engine_dir,
            "rank": 0,  # Single GPU
        }
        
        # For Python session, we need to set these parameters
        # Note: ModelRunner.from_dir will read engine config, but we can override
        runner = ModelRunner.from_dir(**runner_kwargs)
        
        # Update runner parameters if needed
        if hasattr(runner, 'max_output_len'):
            runner.max_output_len = max_output_len
        if hasattr(runner, 'max_input_len'):
            runner.max_input_len = max_input_len
    except Exception as e:
        print(f"Error initializing ModelRunner: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test instances
    test_instances = []
    
    # Prompt-only tests
    for n_p in args.n_prompt:
        if n_p > 0:
            test_instances.append(("prompt", n_p, 0))
    
    # Generation-only tests
    for n_g in args.n_gen:
        if n_g > 0:
            test_instances.append(("generation", 0, n_g))
    
    # Prompt+Generation tests
    if args.prompt_gen:
        for pg_str in args.prompt_gen:
            try:
                n_p, n_g = map(int, pg_str.split(","))
                if n_p > 0 and n_g > 0:
                    test_instances.append(("both", n_p, n_g))
            except:
                print(f"Warning: Invalid prompt-gen pair: {pg_str}")
    
    # Run tests
    results = []
    
    for test_type, n_prompt, n_gen in test_instances:
        print(f"\nRunning test: type={test_type}, prompt={n_prompt}, gen={n_gen}")
        
        # Initialize monitors
        if use_monitoring:
            power_monitor = JetsonPowerMonitor(5)
            temp_monitor = JetsonTempMonitor(1000, is_gpu=True)
            freq_monitor = JetsonFreqMonitor(100)
        else:
            power_monitor = temp_monitor = freq_monitor = None
        
        time_holder = TimeInfoHolder()
        samples_ns = []
        
        # Start monitoring
        if use_monitoring:
            power_monitor.start_monitoring()
            temp_monitor.start_monitoring()
            freq_monitor.start_monitoring()
        
        # Warmup
        if n_prompt > 0 and n_gen == 0:
            test_prompt(runner, tokenizer, min(n_prompt, 32), pad_id, end_id, time_holder, -1)
        elif n_gen > 0 and n_prompt == 0:
            test_generation(runner, tokenizer, 1, 0, pad_id, end_id, time_holder, -1)
        elif n_prompt > 0 and n_gen > 0:
            test_prompt_and_generation(runner, tokenizer, min(n_prompt, 32), 1, pad_id, end_id, time_holder, -1)
        
        time_holder.items.clear()  # Clear warmup data
        
        # Run repetitions
        for rep_id in range(args.repetitions):
            rep_start = get_current_time_ns()
            
            if n_prompt > 0 and n_gen == 0:
                test_prompt(runner, tokenizer, n_prompt, pad_id, end_id, time_holder, rep_id)
            elif n_gen > 0 and n_prompt == 0:
                test_generation(runner, tokenizer, n_gen, 0, pad_id, end_id, time_holder, rep_id)
            elif n_prompt > 0 and n_gen > 0:
                test_prompt_and_generation(runner, tokenizer, n_prompt, n_gen, pad_id, end_id, time_holder, rep_id)
            
            rep_end = get_current_time_ns()
            samples_ns.append(rep_end - rep_start)
        
        # Stop monitoring
        if use_monitoring:
            power_monitor.stop_monitoring()
            temp_monitor.stop_monitoring()
            freq_monitor.stop_monitoring()
        
        # Calculate statistics
        avg_ns = np.mean(samples_ns) if samples_ns else 0
        stddev_ns = np.std(samples_ns) if len(samples_ns) > 1 else 0
        
        n_tokens = n_prompt + n_gen
        avg_ts = (1e9 * n_tokens / avg_ns) if avg_ns > 0 else 0
        stddev_ts = (1e9 * n_tokens * stddev_ns / (avg_ns ** 2)) if avg_ns > 0 else 0
        
        # Save monitoring data
        timestamp = int(time.time())
        data_file = output_dir / f"trt_data_{test_type}_p{n_prompt}_g{n_gen}_{timestamp}.json"
        
        data_map = {
            "time": time_holder.data_to_json_str()
        }
        
        if use_monitoring:
            data_map["power"] = power_monitor.data_to_json_str()
            data_map["temp"] = temp_monitor.data_to_json_str()
            data_map["freq"] = freq_monitor.data_to_json_str()
        
        dump_data_to_json(str(data_file), data_map)
        
        # Create result entry
        result = {
            "test_type": test_type,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_tokens": n_tokens,
            "repetitions": args.repetitions,
            "avg_ns": int(avg_ns),
            "stddev_ns": int(stddev_ns),
            "avg_ts": float(avg_ts),
            "stddev_ts": float(stddev_ts),
            "samples_ns": samples_ns,
            "data_file": str(data_file),
            "engine_dir": args.engine_dir,
            "tokenizer_dir": args.tokenizer_dir,
        }
        
        results.append(result)
        
        # Print summary
        print(f"  Average: {avg_ts:.2f} ± {stddev_ts:.2f} tokens/s")
        print(f"  Data saved to: {data_file}")
    
    # Save summary results
    summary_file = output_dir / f"trt_bench_summary_{int(time.time())}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("Benchmark completed!")


if __name__ == "__main__":
    main()

