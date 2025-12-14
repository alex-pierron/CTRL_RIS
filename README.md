# CTRL_RIS: Consistent and Trustworthy Reinforcement Learning for Reconfigurable Intelligent Surfaces / Investigating AI Vulnerabilities in RL-driven RIS for Beyond-5G and 6G Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

## 🎯 Overview

This repository investigates vulnerabilities, notably fairness, in AI-driven Reconfigurable Intelligent Surfaces (RIS) for Beyond 5G (B5G) and 6G communication frameworks. We focus on identifying, analyzing, and mitigating risks inherent in Reinforcement Learning (RL) control systems to ensure robust and fair implementations for next-generation wireless networks.

### 🔬 Research Focus
- **Fairness Analysis**: Comprehensive evaluation of AI model fairness in RIS control
- **Vulnerability Assessment**: Identification of potential bias and unfairness sources
- **Mitigation Strategies**: Development of robust RL algorithms for fair RIS operation
- **B5G/6G Integration**: Next-generation wireless network optimization

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.9+ recommended)
- **FFMPEG** (for rendering capabilities) - [Download here](https://www.ffmpeg.org/download.html)
- **CUDA** (optional, for GPU acceleration)

### Environment Setup

Choose your preferred setup method:

#### 🐍 Option 1: Conda (Recommended)

Conda provides isolated environments with seamless dependency management:

```bash
# Navigate to the project directory
cd /path/to/CTRL_RIS

# For GPU users (recommended for training):
conda env create -f requirements/GPU_environment.yml
conda activate ctrl_ris_env_GPU

# For CPU users (faster installation):
conda env create -f requirements/CPU_environment.yml
conda activate ctrl_ris_env_CPU
```

#### 🐳 Option 2: Docker (Cross-platform)

For a containerized setup that works on both CPU and GPU:

```bash
# Pull the Docker image
docker pull ap496/ctrl_ris

# Run the Docker container interactively
docker run -it ap496/ctrl_ris
```

#### 🐍 Option 3: Python venv

For a lightweight virtual environment:

```bash
# Create and activate virtual environment
python -m venv ctrl_ris_env

# Activate (choose your platform):
# macOS/Linux:
source ctrl_ris_env/bin/activate

# Windows PowerShell:
.\ctrl_ris_env\Scripts\Activate.ps1

# Windows CMD:
.\ctrl_ris_env\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements/requirements_gpu_environment.txt  # GPU users
# OR
pip install -r requirements/requirements_cpu_environment.txt  # CPU users
```

## 🧪 Running Your First Experiments

### Step 1: Choose Your Configuration

The project includes several pre-configured experiment setups in `src/config_files/`:

- **`Example_1.yaml`** - Basic RIS fairness analysis
- **`Example_2.yaml`** - Advanced multi-user scenarios  
- **`Example_3.yaml`** - Complex B5G/6G network simulation
- **`test.yaml`** - Quick testing configuration

### Step 2: Launch Your Experiment

```bash
# Navigate to the training directory
cd src/train

# Run your experiment (replace 'your_config_file' with actual filename)
python train_RIS.py --config_file Example_1.yaml

# For custom configurations:
python train_RIS.py --config_file your_custom_config.yaml
```

### Step 3: Monitor Results

Your experiment will automatically:
- ✅ **Save training logs** in `data/analytics/paper/`
- 📊 **Generate performance plots** and visualizations
- 🎬 **Create animation files** (if enabled)
- 📈 **Export metrics** for analysis

### 🔧 Customizing Experiments

Create your own configuration by copying an existing YAML file:

```bash
# Copy a template
cp src/config_files/Example_1.yaml src/config_files/my_experiment.yaml

# Edit the configuration
nano src/config_files/my_experiment.yaml  # or use your preferred editor
```

### 📊 Understanding Output

Results are organized as follows:
```
data/analytics/paper/
├── Example_1/
│   ├── ddpg/
│   │   ├── Baseline/          # Baseline algorithm results
│   │   ├── R_fm/             # Fairness metric rewards
│   │   ├── R_fs/             # Fairness smoothed rewards  
│   │   └── R_qos/            # QoS-based rewards
│   └── plots/                # Generated visualizations
└── ...
```

### 🚀 Advanced Usage

For more complex experiments, you can:
- **Modify algorithm parameters** in the YAML files
- **Adjust environment settings** for different scenarios
- **Enable/disable rendering** for faster training
- **Configure multi-processing** for parallel execution
## 📁 Repository Structure

```
CTRL_RIS/
├── 📊 data/                          # Experimental data and results
│   ├── analytics/                    # Analysis results and visualizations
│   ├── archived_experiments/         # Historical experiment data
│   └── raw/                          # Raw experimental outputs
├── 💻 src/                           # Source code
│   ├── 🧠 algorithms/               # Reinforcement Learning algorithms
│   │   ├── ddpg.py                  # Deep Deterministic Policy Gradient
│   │   ├── td3.py                   # Twin Delayed DDPG
│   │   ├── sac.py                   # Soft Actor-Critic
│   │   ├── ppo.py                   # Proximal Policy Optimization
│   │   └── replay_buffer.py         # Experience replay buffer
│   ├── ⚙️ config_files/             # Experiment configurations
│   ├── 🌍 environment/              # RL environment implementation
│   │   ├── RIS_duplex.py           # Main RIS environment
│   │   ├── physics/                 # Signal processing and channel modeling
│   │   ├── rewards/                 # Reward function implementations
│   │   └── multiprocessing/         # Parallel execution support
│   ├── 🏃 runners/                  # Training execution scripts
│   └── 🎯 train/                    # Main training and evaluation scripts
├── 🐳 Dockerfile                    # Container configuration
├── 📋 requirements/                 # Environment specifications
└── 📖 README.md                     # This documentation
```

## 🔍 Key Components

### 🧠 Reinforcement Learning Algorithms

| Algorithm | File | Description | Use Case |
|-----------|------|-------------|----------|
| **DDPG** | `ddpg.py` | Deep Deterministic Policy Gradient for continuous control | Baseline RIS control |
| **TD3** | `td3.py` | Twin Delayed DDPG with improved stability | Enhanced performance |
| **SAC** | `sac.py` | Soft Actor-Critic for sample efficiency | State-of-the-art comparison |
| **PPO** | `ppo.py` | Proximal Policy Optimization | On-policy alternative |
| **Buffer** | `replay_buffer.py` | Experience replay management | All off-policy algorithms |

### 🌍 RIS Environment

- **`RIS_duplex.py`**: Core Gymnasium-based RL environment
  - Full-duplex communication system modeling
  - Base Station (BS) and user interactions
  - RIS-assisted signal quality optimization
  - Multi-user fairness evaluation

- **`physics/signal_processing.py`**: Physics-based simulation
  - Wireless channel modeling
  - Signal propagation calculations
  - Duplex communication processing
  - Realistic B5G/6G network conditions

### ⚙️ Configuration System

- **YAML-based configuration** for all experiment parameters
- **Modular design** allowing easy parameter modification
- **Pre-configured examples** for different scenarios
- **Validation** of configuration parameters

### 🏃 Training Infrastructure

- **Multi-processing support** for parallel execution
- **GPU acceleration** for neural network training
- **Real-time monitoring** and logging
- **Automated result visualization**

## 🤝 Contributing

We welcome contributions from the community to enhance CTRL_RIS's capabilities and robustness. Your contributions help advance research in AI fairness for next-generation wireless networks.

### How to Contribute

1. **🐛 Bug Reports**: Found an issue? Please report it with detailed reproduction steps
2. **📚 Documentation**: Help improve our documentation and examples
3. **🔬 Research**: Submit experimental results and analysis
4. **💡 Features**: Propose new algorithms or environment enhancements
5. **🧪 Testing**: Help improve test coverage and validation

### Development Workflow

```bash
# Fork the repository
git clone https://github.com/alex-pierron/CTRL_RIS.git
cd CTRL_RIS

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and test
python -m pytest tests/

# Submit a pull request
git push origin feature/your-feature-name
```

## 📚 Citation

If you use CTRL_RIS in your research, please cite our work using one of the following formats:

### Software Citation (Recommended)
```bibtex
@software{CTRL_RIS_2025,
  author = {Pierron, Alex and Barbeau, Michel and De Cicco, Luca and Rubio-Hernan, Jose and Garcia-Alfaro, Joaquin},
  title = {CTRL-RIS: Consistent and Trustworthy Reinforcement Learning for Reconfigurable Intelligent Surfaces — Investigating AI Vulnerabilities in RL-driven RIS for Beyond-5G and 6G Systems},
  year = {2025},
  url = {https://github.com/alex-pierron/CTRL_RIS},
  version = {1.0.0},
  license = {GPL-3.0}
}
```

### Traditional Academic Citation
```bibtex
@misc{CTRL_RIS_2025,
  author = {Pierron, Alex and Barbeau, Michel and De Cicco, Luca and Rubio-Hernan, Jose and Garcia-Alfaro, Joaquin},
  title = {CTRL_RIS: Consistent and Trustworthy Reinforcement Learning for Reconfigurable Intelligent Surfaces — Investigating AI Vulnerabilities in RL-driven RIS for Beyond-5G and 6G Systems},
  year = {2025},
  howpublished = {\url{https://github.com/alex-pierron/CTRL_RIS}},
  note = {Available at: \url{https://github.com/alex-pierron/CTRL_RIS}}
}
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


<div align="center">

**⭐ Star this repository if you find it useful!**

[![GitHub stars](https://img.shields.io/github/stars/alex-pierron/CTRL_RIS?style=social)](https://github.com/alex-pierron/CTRL_RIS)
[![GitHub forks](https://img.shields.io/github/forks/alex-pierron/CTRL_RIS?style=social)](https://github.com/alex-pierron/CTRL_RIS)

</div>
