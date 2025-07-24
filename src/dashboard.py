import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List
import logging

from .model_manager import ModelManager
from .benchmark_engine import BenchmarkEngine
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)

class Dashboard:
    """Streamlit dashboard for model benchmarking visualization."""
    
    def __init__(self):
        # Configure page and theme
        st.set_page_config(page_title="AI Model Inference Optimizer", layout="wide")
        
        # Custom theme configuration
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
                
                html, body, [class*="css"] {
                    font-family: 'Poppins', sans-serif;
                }
                
                .stButton > button {
                    background-color: #A2678A;
                    color: white;
                    border-radius: 4px;
                    border: none;
                    padding: 0.5rem 1rem;
                    font-family: 'Poppins', sans-serif;
                }
                
                .stButton > button:hover {
                    background-color: #915c7c;
                }
                
                .streamlit-expanderHeader {
                    font-family: 'Poppins', sans-serif;
                    font-weight: 500;
                }
                
                .css-1d391kg, .css-12oz5g7 {
                    background-color: #F9F6F0;
                }
                
                .css-1n76uvr {
                    background-color: #E8E6E1;
                }
                
                .css-10trblm, .css-1q8dd3e {
                    color: #3E3E3E;
                    font-family: 'Poppins', sans-serif;
                    font-weight: 600;
                }
                
                .css-1kqn0eb {
                    color: #3E3E3E;
                    font-family: 'Poppins', sans-serif;
                }
                
                /* Sidebar styling */
                .css-1v0mbdj {
                    background-color: #E8E6E1;
                    border-right: 1px solid rgba(38, 39, 48, 0.1);
                }
                
                /* Metric containers */
                [data-testid="stMetricValue"] {
                    font-family: 'Poppins', sans-serif;
                    font-weight: 600;
                    color: #A2678A;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Configure theme
        st.markdown("""
            <style>
                :root {
                    --primary-color: #A2678A;
                    --background-color: #F9F6F0;
                    --secondary-background-color: #E8E6E1;
                    --text-color: #3E3E3E;
                }
            </style>
        """, unsafe_allow_html=True)
    def run(self):
        """Run the dashboard application."""
        st.title("AI Model Inference Optimizer Dashboard")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model settings
            st.subheader("Model Settings")
            model_name = st.text_input("Model Name", "bert-base-uncased")
            device = st.selectbox("Device", ["cuda", "cpu"])
            precision = st.multiselect("Precision", ["FP32", "FP16", "INT8"], default=["FP32"])
            
            # Benchmark settings
            st.subheader("Benchmark Settings")
            batch_sizes = st.multiselect("Batch Sizes", [1, 8, 32, 64], default=[1, 8])
            num_iterations = st.number_input("Number of Iterations", min_value=1, value=100)
            
            # Run button
            if st.button("Run Benchmark"):
                self._run_benchmark(
                    model_name, device, precision, batch_sizes, num_iterations
                )
                
        # Main content area
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Latency Analysis")
            self._plot_latency_comparison()
            
        with col2:
            st.header("Memory Usage")
            self._plot_memory_usage()
            
        # Hardware utilization
        st.header("Hardware Utilization")
        self._plot_hardware_utilization()
        
        # Download reports
        st.header("Download Reports")
        self._add_download_buttons()
        
    def _run_benchmark(
        self, 
        model_name: str,
        device: str,
        precision: List[str],
        batch_sizes: List[int],
        num_iterations: int
    ):
        """Run benchmarking with selected configuration."""
        try:
            with st.spinner("Running benchmark..."):
                # Initialize components
                model_manager = ModelManager(model_name, device)
                metrics_tracker = MetricsTracker("logs", device)
                benchmark_engine = BenchmarkEngine(
                    model_manager,
                    metrics_tracker,
                    batch_sizes,
                    precision,
                    num_iterations
                )
                
                # Run benchmarks
                pytorch_results = benchmark_engine.run_pytorch_benchmark()
                
                # Convert to ONNX and run ONNX benchmarks
                model_manager.convert_to_onnx("model.onnx")
                onnx_results = benchmark_engine.run_onnx_benchmark()
                
                # Save results
                self._save_results(pytorch_results, onnx_results)
                
            st.success("Benchmark completed successfully!")
            
        except Exception as e:
            st.error(f"Benchmark failed: {str(e)}")
            logger.error(f"Benchmark error: {str(e)}")
            
    def _plot_latency_comparison(self):
        """Plot latency comparison between PyTorch and ONNX models."""
        try:
            results = self._load_latest_results()
            if not results:
                st.info("No benchmark results available")
                return
                
            fig = go.Figure()
            
            # Add PyTorch results
            pytorch_data = [(k, v["mean_latency"]) for k, v in results.items() if k.startswith("pytorch")]
            if pytorch_data:
                labels, values = zip(*pytorch_data)
                fig.add_bar(
                    name="PyTorch",
                    x=[l.split("_")[1] for l in labels],  # Extract batch size
                    y=values,
                    text=[f"{v:.2f}ms" for v in values]
                )
                
            # Add ONNX results
            onnx_data = [(k, v["mean_latency"]) for k, v in results.items() if k.startswith("onnx")]
            if onnx_data:
                labels, values = zip(*onnx_data)
                fig.add_bar(
                    name="ONNX",
                    x=[l.split("_")[1] for l in labels],  # Extract batch size
                    y=values,
                    text=[f"{v:.2f}ms" for v in values]
                )
                
            fig.update_layout(
                title="Latency Comparison",
                xaxis_title="Batch Size",
                yaxis_title="Latency (ms)",
                barmode="group"
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error plotting latency comparison: {str(e)}")
            
    def _plot_memory_usage(self):
        """Plot memory usage trends."""
        try:
            metrics_df = self._load_metrics()
            if metrics_df is None or metrics_df.empty:
                st.info("No memory usage data available")
                return
                
            fig = go.Figure()
            
            if "ram_usage" in metrics_df.columns:
                fig.add_scatter(
                    name="RAM Usage",
                    x=metrics_df.index,
                    y=metrics_df["ram_usage"],
                    mode="lines"
                )
                
            if "gpu_memory_used" in metrics_df.columns:
                fig.add_scatter(
                    name="GPU Memory",
                    x=metrics_df.index,
                    y=metrics_df["gpu_memory_used"],
                    mode="lines"
                )
                
            fig.update_layout(
                title="Memory Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Memory (MB)"
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error plotting memory usage: {str(e)}")
            
    def _plot_hardware_utilization(self):
        """Plot CPU and GPU utilization."""
        try:
            metrics_df = self._load_metrics()
            if metrics_df is None or metrics_df.empty:
                st.info("No hardware utilization data available")
                return
                
            fig = go.Figure()
            
            if "cpu_percent" in metrics_df.columns:
                fig.add_scatter(
                    name="CPU Utilization",
                    x=metrics_df.index,
                    y=metrics_df["cpu_percent"],
                    mode="lines"
                )
                
            if "gpu_utilization" in metrics_df.columns:
                fig.add_scatter(
                    name="GPU Utilization",
                    x=metrics_df.index,
                    y=metrics_df["gpu_utilization"],
                    mode="lines"
                )
                
            fig.update_layout(
                title="Hardware Utilization Over Time",
                xaxis_title="Time",
                yaxis_title="Utilization (%)"
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error plotting hardware utilization: {str(e)}")
            
    def _add_download_buttons(self):
        """Add buttons to download reports and logs."""
        try:
            if self._has_results():
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Download Benchmark Results (JSON)"):
                        self._download_results()
                        
                with col2:
                    if st.button("Download Metrics Log (CSV)"):
                        self._download_metrics()
                        
        except Exception as e:
            st.error(f"Error setting up download buttons: {str(e)}")
            
    def _save_results(self, pytorch_results: Dict, onnx_results: Dict):
        """Save benchmark results to file."""
        results = {**pytorch_results, **onnx_results}
        with open("reports/latest_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    def _load_latest_results(self) -> Dict:
        """Load the latest benchmark results."""
        try:
            with open("reports/latest_results.json", "r") as f:
                return json.load(f)
        except:
            return {}
            
    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics from CSV file."""
        try:
            return pd.read_csv("logs/metrics.csv", parse_dates=["timestamp"])
        except:
            return None
            
    def _has_results(self) -> bool:
        """Check if benchmark results exist."""
        return (
            Path("reports/latest_results.json").exists() and
            Path("logs/metrics.csv").exists()
        )
        
    def _download_results(self):
        """Provide download link for benchmark results."""
        try:
            with open("reports/latest_results.json", "r") as f:
                st.download_button(
                    "Download Results",
                    f.read(),
                    "benchmark_results.json",
                    "application/json"
                )
        except Exception as e:
            st.error(f"Error downloading results: {str(e)}")
            
    def _download_metrics(self):
        """Provide download link for metrics log."""
        try:
            with open("logs/metrics.csv", "r") as f:
                st.download_button(
                    "Download Metrics",
                    f.read(),
                    "metrics.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"Error downloading metrics: {str(e)}")
            
if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
