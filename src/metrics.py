import psutil
import torch
import json
import csv
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional
import pynvml  # For NVIDIA GPU metrics

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and logs performance metrics during model inference."""
    
    def __init__(self, output_dir: str, device: str = "cuda"):
        """Initialize the metrics tracker.
        
        Args:
            output_dir: Directory to save metric logs
            device: Device being used for inference
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.current_metrics = {}
        
        # Initialize GPU monitoring if using CUDA
        if device == "cuda":
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA GPU monitoring: {str(e)}")
                self.handle = None
                
    def start_iteration(self):
        """Start tracking metrics for a new iteration."""
        self.current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(),
            "ram_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        
        if self.device == "cuda" and self.handle:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.current_metrics.update({
                    "gpu_memory_used": gpu_info.used / 1024 / 1024,  # MB
                    "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {str(e)}")
                
    def end_iteration(self):
        """End tracking metrics for the current iteration."""
        # Update with final CPU/RAM usage
        self.current_metrics.update({
            "final_cpu_percent": psutil.cpu_percent(),
            "final_ram_usage": psutil.Process().memory_info().rss / 1024 / 1024
        })
        
        if self.device == "cuda" and self.handle:
            try:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.current_metrics.update({
                    "final_gpu_memory_used": gpu_info.used / 1024 / 1024,
                    "final_gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                })
            except Exception as e:
                logger.warning(f"Failed to get final GPU metrics: {str(e)}")
                
        self._save_metrics()
        
    def get_metrics(self) -> Dict:
        """Get the current metrics."""
        return self.current_metrics
        
    def _save_metrics(self):
        """Save the current metrics to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        json_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.current_metrics, f, indent=4)
            
        # Save to CSV
        csv_path = self.output_dir / "metrics.csv"
        is_new_file = not csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.current_metrics.keys())
            if is_new_file:
                writer.writeheader()
            writer.writerow(self.current_metrics)
            
    def __del__(self):
        """Cleanup NVIDIA management library."""
        if self.device == "cuda":
            try:
                pynvml.nvmlShutdown()
            except:
                pass
