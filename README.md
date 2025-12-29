# NSKI: Neural Surgical KV-cache Intervention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Publication-grade implementation of Neural Surgical KV-cache Intervention for LLM safety alignment.**

NSKI achieves **O(1) complexity** safety intervention by surgically modifying the KV-cache during inference, projecting out refusal-encoding directions from value representations.

## 🎯 Key Results

| Method | ASR ↓ | Utility | Complexity | Memory Overhead |
|--------|-------|---------|------------|-----------------|
| Baseline | 46% | 100% | - | - |
| Arditi et al. (2024) | 15% | 95% | O(T) | Moderate |
| Belitsky et al. (2025) | 18% | 90% | O(T) | High |
| **NSKI (Ours)** | **5%** | **100%** | **O(1)** | **None** |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pranjaymalhotra/NSKI.git
cd NSKI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets
python -m nski.data.download
```

### Run Experiments

```bash
# Full experimental suite
python -m nski.experiments.run_all

# Individual experiments
python -m nski.experiments.main_comparison
python -m nski.experiments.ablation_study
python -m nski.experiments.baseline_comparison
```

## 📁 Project Structure

```
NSKI/
├── nski/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── kv_hooks.py          # Real KV-cache hooks (CRITICAL)
│   │   ├── refusal_direction.py # Refusal direction extraction
│   │   ├── surgery.py           # Surgical intervention
│   │   └── utils.py             # Utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py            # Multi-model loader
│   │   └── supported.py         # Supported model configs
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── arditi.py            # Arditi et al. (2024)
│   │   ├── belitsky.py          # Belitsky et al. (2025)
│   │   └── jbshield.py          # JBSHIELD
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # Dataset downloader
│   │   ├── advbench.py          # AdvBench (520 prompts)
│   │   ├── alpaca.py            # Alpaca (harmless)
│   │   └── harmbench.py         # HarmBench (adversarial)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # ASR, utility, perplexity
│   │   ├── judges.py            # Refusal detection
│   │   └── statistical.py       # Bootstrap, CI, effect size
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_all.py           # Master experiment runner
│   │   ├── main_comparison.py   # NSKI vs baselines
│   │   ├── ablation_study.py    # Hyperparameter ablations
│   │   └── adversarial.py       # GCG, AutoPrompt robustness
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # Publication figures
├── configs/
│   └── default.yaml             # Default configuration
├── results/                     # Experiment outputs
├── figures/                     # Generated plots
├── requirements.txt
├── setup.py
└── README.md
```

## 🔬 Method Overview

### NSKI Algorithm

1. **Extract Refusal Direction**: Compute mean activation difference between harmful and harmless prompts at target layer
2. **Register KV Hook**: Attach forward hook to target attention layer
3. **Surgical Intervention**: Project out refusal direction from value representations
4. **Generate**: Run normal inference with modified KV-cache

```python
# Simplified NSKI intervention
V_modified = V - (V @ refusal_direction) @ refusal_direction.T
```

### Why O(1)?

Unlike activation steering methods that modify every token's representation (O(T) complexity), NSKI performs a **single operation** on the KV-cache that persists through generation.

## 📊 Supported Models

| Model | Parameters | Status |
|-------|------------|--------|
| Llama-3-8B-Instruct | 8B | ✅ Primary |
| Llama-2-7B-Chat | 7B | ✅ Tested |
| Mistral-7B-Instruct | 7B | ✅ Tested |
| GPT-2-XL | 1.5B | ✅ Tested |
| Phi-3-Mini | 3.8B | ✅ Tested |

## 📈 Benchmarks

- **AdvBench**: 520 harmful prompts (Zou et al., 2023)
- **HarmBench**: Adversarial prompts (Mazeika et al., 2024)
- **Alpaca**: 200 harmless instructions (Taori et al., 2023)
- **WikiText-103**: Perplexity evaluation

## 🧪 Baselines Implemented

1. **Arditi et al. (2024)**: Residual stream steering
2. **Belitsky et al. (2025)**: Attention head modulation  
3. **JBSHIELD (2024)**: Jailbreak defense via prompt filtering

## 📖 Citation

```bibtex
@article{malhotra2025nski,
  title={NSKI: Neural Surgical KV-cache Intervention for LLM Safety},
  author={Malhotra, Pranjay},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- AdvBench dataset from Zou et al. (2023)
- Inspiration from Arditi et al. (2024) and Belitsky et al. (2025)
