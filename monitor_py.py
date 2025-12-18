#!/usr/bin/env python3
"""
Python version of monitor.h for TensorRT-LLM benchmarking
Provides power, temperature, and frequency monitoring on Jetson devices
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


def is_jetson() -> bool:
    """Check if running on a Jetson device"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            if "NVIDIA" in model or "Jetson" in model:
                return True
    except:
        pass
    
    # Fallback: check nvidia-smi
    if os.system("nvidia-smi > /dev/null 2>&1") == 0:
        return True
    
    return False


def read_value_from_file(path: str) -> float:
    """Read a numeric value from a file"""
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except:
        return 0.0


def create_dir_if_not_exists(file_path: str):
    """Create directory if it doesn't exist"""
    dir_path = Path(file_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class TimeInfoItem:
    """Time information for a single inference call"""
    n_p_eval: int  # number of tokens in eval calls for the prompt
    n_eval: int    # number of eval calls
    rep_id: int    # repetition id
    start_time_ns: int
    end_time_ns: int


class TimeInfoHolder:
    """Holder for time information"""
    
    def __init__(self):
        self.items: List[TimeInfoItem] = []
    
    def add_info(self, item: TimeInfoItem):
        """Add a time info item"""
        self.items.append(item)
    
    def data_to_json_str(self) -> str:
        """Convert to JSON string"""
        json_list = []
        for item in self.items:
            json_list.append({
                "n_p_eval": item.n_p_eval,
                "n_eval": item.n_eval,
                "rep_id": item.rep_id,
                "start_time_ns": item.start_time_ns,
                "end_time_ns": item.end_time_ns
            })
        return json.dumps(json_list, indent=2)


@dataclass
class MonitoringData:
    """Monitoring data for a single monitoring session"""
    sampling_duration_ms: int
    data_type: str  # "current", "temperature", "frequency"
    module_data_map: Dict[str, List[float]]
    timestamps: Dict[str, List[int]]
    start_time_ns: int
    end_time_ns: int
    
    def add_module_data(self, module_name: str, value: float):
        """Add data for a module"""
        if module_name not in self.module_data_map:
            self.module_data_map[module_name] = []
            self.timestamps[module_name] = []
        
        self.module_data_map[module_name].append(value)
        self.timestamps[module_name].append(int(time.time_ns()))
    
    def data_to_json_str(self) -> str:
        """Convert to JSON string"""
        result = {
            "sampling_duration_ms": self.sampling_duration_ms,
            self.data_type: {},
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ns": self.end_time_ns - self.start_time_ns
        }
        
        for module_name, values in self.module_data_map.items():
            result[self.data_type][module_name] = {
                "timestamps": self.timestamps[module_name],
                "value": values
            }
        
        return json.dumps(result, indent=2)


class BaseMonitor:
    """Base class for monitoring"""
    
    def __init__(self, sampling_duration_ms: int):
        self.sampling_duration_ms = sampling_duration_ms
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.md_array: List[MonitoringData] = []
        self.lock = threading.Lock()
    
    def start_monitoring(self, sampling_duration_ms: Optional[int] = None):
        """Start monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        duration = sampling_duration_ms if sampling_duration_ms is not None else self.sampling_duration_ms
        
        md = MonitoringData(
            sampling_duration_ms=duration,
            data_type=self.get_data_type(),
            module_data_map={},
            timestamps={},
            start_time_ns=int(time.time_ns()),
            end_time_ns=0
        )
        
        self.md_array.append(md)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(md,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.monitoring or not self.monitor_thread:
            return
        
        with self.lock:
            self.monitoring = False
        
        self.monitor_thread.join()
        
        # Set end time for the last monitoring data
        if self.md_array:
            self.md_array[-1].end_time_ns = int(time.time_ns())
    
    def clear_data(self):
        """Clear all monitoring data"""
        self.md_array.clear()
    
    def get_monitoring_data(self) -> List[MonitoringData]:
        """Get all monitoring data"""
        return self.md_array
    
    def data_to_json_str(self) -> str:
        """Convert all monitoring data to JSON string"""
        json_list = []
        for md in self.md_array:
            json_list.append(json.loads(md.data_to_json_str()))
        return json.dumps(json_list, indent=2)
    
    def dump_data_to_json(self, file_path: str):
        """Dump data to JSON file"""
        create_dir_if_not_exists(file_path)
        with open(file_path, "w") as f:
            f.write(self.data_to_json_str())
    
    def _monitor_loop(self, md: MonitoringData):
        """Monitoring loop"""
        count = 0
        sampling_start_time = time.time_ns()
        
        while self.monitoring:
            self.fill_data(md)
            count += 1
            
            next_sample_time = sampling_start_time + count * self.sampling_duration_ms * 1_000_000
            sleep_time = (next_sample_time - time.time_ns()) / 1_000_000_000
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            with self.lock:
                if not self.monitoring:
                    break
        
        md.end_time_ns = int(time.time_ns())
    
    def fill_data(self, md: MonitoringData):
        """Fill monitoring data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_data_type(self) -> str:
        """Get data type - to be implemented by subclasses"""
        raise NotImplementedError


class JetsonPowerMonitor(BaseMonitor):
    """Monitor power consumption on Jetson"""
    
    def __init__(self, sampling_duration_ms: int = 5):
        super().__init__(sampling_duration_ms)
        self.channel_paths = {
            "VDD_GPU_SOC": "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input",
            "VDD_CPU_CV": "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr2_input",
            "VIN_SYS_5V0": "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr3_input",
        }
    
    def fill_data(self, md: MonitoringData):
        """Read power data"""
        for module_name, path in self.channel_paths.items():
            current = read_value_from_file(path) / 1000.0  # Convert mA to A
            md.add_module_data(module_name, current)
    
    def get_data_type(self) -> str:
        return "current"


class JetsonTempMonitor(BaseMonitor):
    """Monitor temperature on Jetson"""
    
    def __init__(self, sampling_duration_ms: int = 1000, is_gpu: bool = True):
        super().__init__(sampling_duration_ms)
        self.is_gpu = is_gpu
    
    def fill_data(self, md: MonitoringData):
        """Read temperature data"""
        if self.is_gpu:
            temp_path = "/sys/devices/virtual/thermal/thermal_zone1/temp"
            module_name = "GPU"
        else:
            temp_path = "/sys/devices/virtual/thermal/thermal_zone0/temp"
            module_name = "CPU"
        
        temp = read_value_from_file(temp_path) / 1000.0  # Convert to Celsius
        md.add_module_data(module_name, temp)
    
    def get_data_type(self) -> str:
        return "temperature"


class JetsonFreqMonitor(BaseMonitor):
    """Monitor frequency on Jetson"""
    
    def __init__(self, sampling_duration_ms: int = 100):
        super().__init__(sampling_duration_ms)
        self.freq_paths = {
            "CPU0": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
            "CPU4": "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq",
            "CPU8": "/sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq",
            "GPU": "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq",
            "EMC": "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
        }
    
    def fill_data(self, md: MonitoringData):
        """Read frequency data"""
        for module_name, path in self.freq_paths.items():
            freq = read_value_from_file(path)  # Frequency in Hz
            md.add_module_data(module_name, freq)
    
    def get_data_type(self) -> str:
        return "frequency"


def dump_data_to_json(file_path: str, data_map: Dict[str, str]):
    """Dump multiple data sources to a single JSON file"""
    create_dir_if_not_exists(file_path)
    
    result = {}
    for key, json_str in data_map.items():
        result[key] = json.loads(json_str)
    
    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)

