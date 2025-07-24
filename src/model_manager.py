from typing import Optional, Union, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnx
import onnxruntime as ort
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages the loading, conversion and handling of transformer models."""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """Initialize the model manager.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to load the model on ("cuda" or "cpu")
        """
        self.model_name = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.onnx_path = None
        
    def load_pytorch_model(self) -> None:
        """Load a PyTorch model from HuggingFace."""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"Successfully loaded PyTorch model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise
            
    def convert_to_onnx(self, output_path: str, input_shape: tuple = (1, 512)) -> str:
        """Convert PyTorch model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            input_shape: Input shape for the model
            
        Returns:
            Path to the saved ONNX model
        """
        if self.model is None:
            raise ValueError("PyTorch model must be loaded first")
            
        try:
            dummy_input = torch.randint(0, 100, input_shape).to(self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}},
                do_constant_folding=True,
                opset_version=12
            )
            
            # Verify the ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            self.onnx_path = output_path
            logger.info(f"Successfully converted and saved ONNX model to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert model to ONNX: {str(e)}")
            raise
            
    def load_onnx_model(self, onnx_path: Optional[str] = None) -> None:
        """Load an ONNX model for inference.
        
        Args:
            onnx_path: Path to the ONNX model. If None, uses the previously converted model.
        """
        try:
            path = onnx_path or self.onnx_path
            if path is None:
                raise ValueError("No ONNX model path provided")
                
            # Create ONNX Runtime session
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            self.onnx_session = ort.InferenceSession(path, providers=providers)
            logger.info(f"Successfully loaded ONNX model from: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise
