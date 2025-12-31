"""
Helper script to verify NSKI installation.
"""
import sys
sys.path.insert(0, '.')

print()
print('Checking imports...')

errors = []

try:
    from nski.core import KVCacheHook, RefusalDirectionExtractor, NSKISurgeon
    print('  + Core modules')
except Exception as e:
    errors.append(f'Core: {e}')
    print(f'  - Core modules: {e}')

try:
    from nski.models import ModelLoader, get_model_config, SUPPORTED_MODELS
    print('  + Model modules')
except Exception as e:
    errors.append(f'Models: {e}')
    print(f'  - Model modules: {e}')

try:
    from nski.data import download_advbench, AdvBenchDataset, AlpacaDataset
    print('  + Data modules')
except Exception as e:
    errors.append(f'Data: {e}')
    print(f'  - Data modules: {e}')

try:
    from nski.evaluation import compute_asr, compute_perplexity, KeywordRefusalJudge
    print('  + Evaluation modules')
except Exception as e:
    errors.append(f'Evaluation: {e}')
    print(f'  - Evaluation modules: {e}')

try:
    from nski.baselines import ArditiSteering, BelitskyModulation, JBShield
    print('  + Baseline modules')
except Exception as e:
    errors.append(f'Baselines: {e}')
    print(f'  - Baseline modules: {e}')

try:
    import torch
    print(f'  + PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  + CUDA: {torch.cuda.get_device_name(0)}')
        print(f'  + VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('  ! CUDA not available (CPU mode)')
except Exception as e:
    errors.append(f'PyTorch: {e}')
    print(f'  - PyTorch: {e}')

print()
if errors:
    print('! Setup completed with warnings. Some features may not work.')
else:
    print('+ All checks passed! NSKI is ready.')
