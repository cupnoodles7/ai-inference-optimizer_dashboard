import logging
from pathlib import Path
from src.model_manager import ModelManager
from src.benchmark_engine import BenchmarkEngine
from src.metrics import MetricsTracker
from src.config import ConfigManager
import argparse

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimizer.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for the optimizer."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.parse_cli_args()
    
    try:
        # Initialize components
        model_manager = ModelManager(
            config['model']['name'],
            config['model']['device']
        )
        
        metrics_tracker = MetricsTracker(
            config['output']['log_dir'],
            config['model']['device']
        )
        
        benchmark_engine = BenchmarkEngine(
            model_manager,
            metrics_tracker,
            config['benchmark']['batch_sizes'],
            [config['model']['precision']],
            config['benchmark']['num_iterations']
        )
        
        # Run benchmarks
        logger.info("Starting PyTorch benchmarks...")
        pytorch_results = benchmark_engine.run_pytorch_benchmark()
        
        logger.info("Converting to ONNX...")
        onnx_path = model_manager.convert_to_onnx(
            str(Path(config['output']['report_dir']) / "model.onnx")
        )
        
        logger.info("Starting ONNX benchmarks...")
        onnx_results = benchmark_engine.run_onnx_benchmark()
        
        # Save results
        results_path = Path(config['output']['report_dir']) / "results.json"
        results = {**pytorch_results, **onnx_results}
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
