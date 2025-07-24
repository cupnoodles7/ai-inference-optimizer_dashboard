from typing import List, Dict, Optional
import torch
import numpy as np
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from .model_manager import ModelManager
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)

class BenchmarkEngine:
    """Handles model inference benchmarking with different configurations."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        metrics_tracker: MetricsTracker,
        batch_sizes: List[int] = [1, 8, 32, 64],
        precision_types: List[str] = ["FP32", "FP16"],
        num_iterations: int = 100
    ):
        """Initialize the benchmark engine.
        
        Args:
            model_manager: Instance of ModelManager
            metrics_tracker: Instance of MetricsTracker
            batch_sizes: List of batch sizes to test
            precision_types: List of precision types to test
            num_iterations: Number of iterations per configuration
        """
        self.model_manager = model_manager
        self.metrics_tracker = metrics_tracker
        self.batch_sizes = batch_sizes
        self.precision_types = precision_types
        self.num_iterations = num_iterations
        
    def run_pytorch_benchmark(self) -> Dict:
        """Run benchmarks for PyTorch model."""
        results = {}
        
        for batch_size in self.batch_sizes:
            for precision in self.precision_types:
                logger.info(f"Running PyTorch benchmark with batch_size={batch_size}, precision={precision}")
                
                # Set precision
                if precision == "FP16" and self.model_manager.device == "cuda":
                    self.model_manager.model.half()
                
                # Generate dummy input
                dummy_input = torch.randint(0, 100, (batch_size, 512)).to(self.model_manager.device)
                if precision == "FP16":
                    dummy_input = dummy_input.half()
                
                # Warmup
                for _ in range(10):
                    with torch.no_grad():
                        _ = self.model_manager.model(dummy_input)
                
                # Benchmark
                latencies = []
                try:
                    for _ in range(self.num_iterations):
                        self.metrics_tracker.start_iteration()
                        
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model_manager.model(dummy_input)
                        latency = (time.perf_counter() - start_time) * 1000  # ms
                        
                        self.metrics_tracker.end_iteration()
                        latencies.append(latency)
                        
                    results[f"pytorch_b{batch_size}_{precision}"] = {
                        "mean_latency": np.mean(latencies),
                        "std_latency": np.std(latencies),
                        "metrics": self.metrics_tracker.get_metrics()
                    }
                    
                except Exception as e:
                    logger.error(f"Error during PyTorch benchmark: {str(e)}")
                    continue
                
        return results
    
    def run_onnx_benchmark(self) -> Dict:
        """Run benchmarks for ONNX model."""
        results = {}
        
        for batch_size in self.batch_sizes:
            logger.info(f"Running ONNX benchmark with batch_size={batch_size}")
            
            # Generate dummy input
            dummy_input = np.random.randint(0, 100, (batch_size, 512)).astype(np.float32)
            ort_inputs = {self.model_manager.onnx_session.get_inputs()[0].name: dummy_input}
            
            # Warmup
            for _ in range(10):
                _ = self.model_manager.onnx_session.run(None, ort_inputs)
            
            # Benchmark
            latencies = []
            try:
                for _ in range(self.num_iterations):
                    self.metrics_tracker.start_iteration()
                    
                    start_time = time.perf_counter()
                    _ = self.model_manager.onnx_session.run(None, ort_inputs)
                    latency = (time.perf_counter() - start_time) * 1000  # ms
                    
                    self.metrics_tracker.end_iteration()
                    latencies.append(latency)
                    
                results[f"onnx_b{batch_size}"] = {
                    "mean_latency": np.mean(latencies),
                    "std_latency": np.std(latencies),
                    "metrics": self.metrics_tracker.get_metrics()
                }
                
            except Exception as e:
                logger.error(f"Error during ONNX benchmark: {str(e)}")
                continue
                
        return results
