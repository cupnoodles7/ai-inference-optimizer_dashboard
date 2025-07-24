import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from src.model_manager import ModelManager
from src.benchmark_engine import BenchmarkEngine
from src.metrics import MetricsTracker

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def model_manager():
    """Create a model manager instance."""
    return ModelManager("bert-base-uncased", device="cpu")

@pytest.fixture
def metrics_tracker(temp_dir):
    """Create a metrics tracker instance."""
    return MetricsTracker(temp_dir, device="cpu")

def test_model_loading(model_manager):
    """Test loading a PyTorch model."""
    model_manager.load_pytorch_model()
    assert model_manager.model is not None
    assert model_manager.tokenizer is not None

def test_onnx_conversion(model_manager, temp_dir):
    """Test converting PyTorch model to ONNX."""
    model_manager.load_pytorch_model()
    onnx_path = str(Path(temp_dir) / "model.onnx")
    converted_path = model_manager.convert_to_onnx(onnx_path)
    assert Path(converted_path).exists()

def test_metrics_tracking(metrics_tracker):
    """Test metrics tracking functionality."""
    metrics_tracker.start_iteration()
    initial_metrics = metrics_tracker.get_metrics()
    assert "cpu_percent" in initial_metrics
    assert "ram_usage" in initial_metrics
    
    metrics_tracker.end_iteration()
    final_metrics = metrics_tracker.get_metrics()
    assert "final_cpu_percent" in final_metrics
    assert "final_ram_usage" in final_metrics

def test_benchmark_execution(model_manager, metrics_tracker):
    """Test benchmark engine execution."""
    model_manager.load_pytorch_model()
    benchmark_engine = BenchmarkEngine(
        model_manager,
        metrics_tracker,
        batch_sizes=[1],
        precision_types=["FP32"],
        num_iterations=2
    )
    
    results = benchmark_engine.run_pytorch_benchmark()
    assert "pytorch_b1_FP32" in results
    assert "mean_latency" in results["pytorch_b1_FP32"]
    assert "metrics" in results["pytorch_b1_FP32"]
