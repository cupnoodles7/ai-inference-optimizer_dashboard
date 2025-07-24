import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List
import logging
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import ModelManager
from benchmark_engine import BenchmarkEngine
from metrics import MetricsTracker

# Custom CSS with Poppins font and enhanced styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: #4A3B43;
        }

        /* Main Headers */
        h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: #A2678A;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }

        /* Subheaders */
        h2 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #865B73;
            font-size: 1.8rem;
            margin-top: 1.5rem;
            letter-spacing: -0.01em;
        }

        /* Section headers */
        h3 {
            font-family: 'Poppins', sans-serif;
            font-weight: 500;
            color: #73515F;
            font-size: 1.4rem;
            margin-top: 1.2rem;
        }

        /* Smaller headers */
        h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
            font-weight: 500;
            color: #624551;
            font-size: 1.1rem;
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Poppins', sans-serif;
            font-weight: 500;
            background-color: #A2678A;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(162, 103, 138, 0.2);
        }

        .stButton > button:hover {
            background-color: #865B73;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(162, 103, 138, 0.3);
        }

        /* Input fields */
        .stSelectbox, .stMultiSelect {
            background-color: #F4ECF0;
            border-radius: 0.5rem;
            border: 1px solid #E4D6DD;
        }

        /* Text and paragraphs */
        .stMarkdown {
            font-family: 'Poppins', sans-serif;
            color: #4A3B43;
            line-height: 1.6;
            font-size: 1rem;
        }

        /* Charts */
        .stPlotlyChart {
            font-family: 'Poppins', sans-serif;
            background-color: #F4ECF0;
            border-radius: 0.75rem;
            padding: 1.2rem;
            box-shadow: 0 4px 6px rgba(162, 103, 138, 0.1);
            border: 1px solid #E4D6DD;
        }

        /* Top decoration bar */
        div[data-testid="stDecoration"] {
            background-image: linear-gradient(90deg, #A2678A, #865B73);
            height: 0.3rem !important;
        }

        /* Toolbar */
        div[data-testid="stToolbar"] {
            background-color: #F4ECF0;
        }

        /* Alerts and info boxes */
        .stAlert {
            background-color: #F4ECF0;
            border: 2px solid #A2678A;
            border-radius: 0.5rem;
            color: #4A3B43;
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: #E4D6DD;
        }

        /* Widgets */
        .stNumberInput, .stTextInput {
            background-color: #F4ECF0;
            border: 1px solid #E4D6DD;
            border-radius: 0.5rem;
            color: #4A3B43;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background-color: #A2678A;
        }

        /* Tables */
        .stTable {
            background-color: #F4ECF0;
            border-radius: 0.5rem;
            border: 1px solid #E4D6DD;
        }

        /* Code blocks */
        .stCodeBlock {
            background-color: #E4D6DD;
            border-radius: 0.5rem;
            border: 1px solid #D6C3CC;
        }
    </style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

class Dashboard:
    """Streamlit dashboard for model benchmarking visualization."""
    
    def __init__(self):
        st.set_page_config(page_title="AI Model Inference Optimizer", layout="wide")
        
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
