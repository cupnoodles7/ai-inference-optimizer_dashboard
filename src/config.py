import yaml
import argparse
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG = {
        "model": {
            "name": "bert-base-uncased",
            "device": "cuda",
            "precision": "FP32"
        },
        "benchmark": {
            "batch_sizes": [1, 8, 32, 64],
            "num_iterations": 100,
            "warmup_iterations": 10
        },
        "output": {
            "log_dir": "logs",
            "report_dir": "reports"
        }
    }
    
    def __init__(self):
        self.config = {}
        
    def load_yaml(self, yaml_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded config from: {yaml_path}")
            self.config = self._merge_with_defaults(config)
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config from {yaml_path}: {str(e)}")
            raise
            
    def parse_cli_args(self) -> Dict:
        """Parse command line arguments.
        
        Returns:
            Dictionary of parsed arguments
        """
        parser = argparse.ArgumentParser(description='AI Model Inference Optimizer')
        
        # Model arguments
        parser.add_argument('--model', type=str, help='Model name or path')
        parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to run on')
        parser.add_argument('--precision', choices=['FP32', 'FP16', 'INT8'], help='Model precision')
        
        # Benchmark arguments
        parser.add_argument('--batch-sizes', type=int, nargs='+', help='Batch sizes to test')
        parser.add_argument('--iterations', type=int, help='Number of iterations')
        
        # Output arguments
        parser.add_argument('--log-dir', type=str, help='Directory for logs')
        parser.add_argument('--config', type=str, help='Path to YAML config file')
        
        args = parser.parse_args()
        cli_config = {k: v for k, v in vars(args).items() if v is not None}
        
        if args.config:
            self.load_yaml(args.config)
            
        self.config = self._merge_with_defaults(cli_config)
        return self.config
        
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge provided configuration with defaults.
        
        Args:
            config: Configuration to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = self.DEFAULT_CONFIG.copy()
        
        # Merge model config
        if 'model' in config:
            merged['model'].update(config['model'])
            
        # Merge benchmark config
        if 'benchmark' in config:
            merged['benchmark'].update(config['benchmark'])
            
        # Merge output config
        if 'output' in config:
            merged['output'].update(config['output'])
            
        return merged
        
    def validate_config(self) -> bool:
        """Validate the current configuration.
        
        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        try:
            # Validate model configuration
            if not self.config['model']['name']:
                raise ValueError("Model name must be specified")
                
            # Validate device
            if self.config['model']['device'] not in ['cuda', 'cpu']:
                raise ValueError("Device must be either 'cuda' or 'cpu'")
                
            # Validate precision
            if self.config['model']['precision'] not in ['FP32', 'FP16', 'INT8']:
                raise ValueError("Precision must be one of: FP32, FP16, INT8")
                
            # Validate batch sizes
            if not all(isinstance(b, int) and b > 0 for b in self.config['benchmark']['batch_sizes']):
                raise ValueError("Batch sizes must be positive integers")
                
            # Validate iterations
            if self.config['benchmark']['num_iterations'] <= 0:
                raise ValueError("Number of iterations must be positive")
                
            # Create output directories
            Path(self.config['output']['log_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['output']['report_dir']).mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
