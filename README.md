### Edge-Vision: TinyML-Powered Anomaly Detection
https://img.shields.io/badge/TinyML-Edge%2520Computing-blue
https://img.shields.io/badge/Privacy-Preserving-green
https://img.shields.io/badge/Python-3.8%252B-yellow
https://img.shields.io/badge/License-Apache%25202.0-orange
Edge-Vision: TinyML-Powered Anomaly Detection

⚡ On-device anomaly detection — Deploying efficient, privacy-preserving computer vision models for smart infrastructure, IoT, and Industry 4.0 applications.

🌍 Background & Motivation
Modern urban environments and industrial facilities generate enormous volumes of visual data through distributed IoT sensors and CCTV networks. Traditional cloud-based processing approaches face significant challenges:
❌ High latency impedes real-time response capabilities

❌ Substantial bandwidth consumption creates network bottlenecks

❌ Energy-intensive processing contradicts sustainability goals

❌ Privacy vulnerabilities from transmitting sensitive visual data

Our solution: Leverage TinyML to perform intelligent anomaly detection directly on edge devices, enabling:
✅ Real-time inference with minimal latency

✅ Significant bandwidth and energy conservation

✅ Enhanced privacy through on-device data processing

✅ Reduced operational costs for large-scale deployments

This project implements a highly optimized, quantized convolutional neural network distilled from a vision transformer backbone, deployable on resource-constrained hardware (Jetson Nano, Coral TPU, Raspberry Pi). The system detects anomalies—from manufacturing defects to security incidents—while preserving privacy through integrated on-device blurring of sensitive regions.

✨ Key Features :

📊 Standardized Benchmarking: Utilizes the MVTec-AD dataset for industrial anomaly detection

🧠 Advanced Architectures: Implements and compares AutoEncoder (AE), PatchCore, and SimCLR approaches

🔄 Model Compression: Employs knowledge distillation from large transformer to compact CNN

📉 Hardware Optimization: Applies quantization-aware training and pruning for edge deployment

🔒 Privacy by Design: Integrates on-device sensitive region blurring before inference

⚡ Performance Analysis: Comprehensive evaluation of FPS/latency vs. accuracy vs. energy consumption

🎥 Real-time Demonstration: Complete pipeline showcasing live anomaly detection with visual overlays

Project Structure :
edge_vision_anomaly_detection/
│
├── data/                           # Dataset and sample assets
│   ├── mvtec/                      # MVTec-AD dataset (optional)
│   └── sample_video.mp4            # Demonstration video
│
├── models/                         # Model definitions and weights
│   ├── baseline_ae.pth             # Autoencoder baseline
│   ├── distilled_cnn.pth           # Distilled compact model
│   └── quantized_cnn.tflite        # Quantized model for deployment
│
├── results/                        # Output directory
│   ├── training_metrics/           # Training logs and metrics
│   ├── performance/                # Hardware performance results
│   │   ├── accuracy_vs_latency.png
│   │   ├── energy_profile.png
│   │   └── hardware_benchmark.csv
│   └── demonstrations/             # Inference examples
│       └── anomaly_detection_demo.mp4
│
├── src/                            # Source code
│   ├── data_processing/            # Data utilities
│   │   ├── mvtec_dataset.py        # MVTec-AD dataset handler
│   │   └── privacy_preprocessor.py # Privacy preservation modules
│   ├── models/                     # Model architectures
│   │   ├── autoencoder.py          # AE implementation
│   │   ├── patchcore.py            # PatchCore implementation
│   │   └── knowledge_distillation.py # Distillation training
│   ├── training/                   # Training routines
│   │   └── train_utils.py          # Training utilities
│   ├── evaluation/                 # Evaluation metrics
│   │   └── eval_mvtec.py           # MVTec evaluation protocol
│   ├── inference/                  # Deployment modules
│   │   ├── edge_inferencer.py      # Hardware-specific inference
│   │   └── anomaly_visualizer.py   # Visualization utilities
│   ├── utils/                      # Helper functions
│   │   ├── config.py               # Configuration management
│   │   └── logger.py               # Logging utilities
│   ├── main.py                     # Main training/evaluation script
│   └── demo.py                     # Demonstration pipeline
│
├── tests/                          # Unit tests
├── docs/                           # Additional documentation
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment setup
└── README.md                       # This document

⚙️ Installation & Setup
Option 1: Pip Installation 
# Clone repository
git clone https://github.com/srizoni-maity/Edge-Vision-Anomaly-Detection-with-TinyML-Privacy-Preserving-AI-for-Smart-Cities-IoT
cd edge_vision_anomaly_detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

Option 2: Conda Installation
# Create conda environment
conda env create -f environment.yml
conda activate edge-vision

Hardware-Specific Setup
For deployment on specific edge devices, additional setup may be required:

NVIDIA Jetson Nano:
# Install JetPack components
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow-gpu
Coral TPU:
# Install Edge TPU runtime
sudo apt-get install edgetpu-compiler python3-pycoral
🚀 Usage Examples

Quick Start with Synthetic Data :
# Train a compact autoencoder on synthetic data
python src/main.py --mode quick_train --epochs 30 --output_dir results/quick_run

# Evaluate model performance
python src/main.py --mode evaluate --model_path results/quick_run/tiny_ae.pth

Generate Demonstration Video :
# Create sample video with synthetic anomalies
python src/utils/create_sample_video.py --output data/sample_video.mp4

# Run inference on sample video
python src/demo.py --video data/sample_video.mp4 --model models/quantized_cnn.tflite --output results/demo_output.mp4

Full Training on MVTec-AD (Requires Dataset)
# Train on specific MVTec category
python src/main.py --mode train --dataset_path data/mvtec --category bottle --epochs 100 --batch_size 32

# Knowledge distillation from teacher to student model
python src/training/knowledge_distillation.py --teacher models/patchcore.pth --student models/tiny_cnn.pth --dataset data/mvtec/hazelnut

Hardware Benchmarking :
# Benchmark model on available hardware
python src/inference/edge_inferencer.py --model models/quantized_cnn.tflite --benchmark --runs 1000

# Compare energy consumption across devices
python src/utils/energy_profiler.py --model models/quantized_cnn.tflite --duration 60
📊 Performance Results
Our optimized models achieve compelling performance trade-offs:

Model	Size (MB)	Accuracy (AUC)	Latency (ms)	Energy (mJ/inf)
Baseline AutoEncoder	18.2	0.87	42.3	125.6
Distilled CNN	4.7	0.85	18.7	63.2
Quantized CNN (INT8)	1.2	0.83	8.4	28.9
*Results measured on Jetson Nano 4GB with MVTec-AD bottle category*

https://results/performance/accuracy_vs_latency.png
Trade-off between detection accuracy and inference latency


🌐 Application Scenarios
🏭 Smart Manufacturing
Real-time defect detection on production lines without cloud dependency, enabling immediate quality control interventions.

🏙️ Urban Surveillance
Privacy-aware monitoring of public spaces with local processing that only transmits alerts rather than continuous footage.

🏥 Healthcare IoT
On-device analysis of medical imaging streams while maintaining strict patient privacy through localized processing.

🔒 Privacy-Sensitive Environments
Edge deployment in settings where data cannot leave the premises due to regulatory or security constraints.

Example Output Visualization
https://results/demonstrations/anomaly_detection_demo.gif
Red-highlighted regions indicate detected anomalies with confidence scores

🔮 Future Enhancements
Hardware Acceleration: Integration with ONNX Runtime and TensorRT for maximum inference speed
Multi-Modal Learning: Fusion of visual data with complementary sensor inputs (thermal, depth, etc.)
Adaptive Learning: Continuous on-device model refinement based on new data patterns
Federated Learning: Privacy-preserving collaborative model improvement across edge devices
Extended Benchmarking: Support for additional anomaly detection datasets and scenarios

👥 Authors
Srizoni Maity & Baishakhi Sing 
Project Development & Research


🙏 Acknowledgments
We thank the creators of the MVTec-AD dataset for providing a standardized benchmark for anomaly detection research.



