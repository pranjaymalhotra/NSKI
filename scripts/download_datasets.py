"""
Helper script to download NSKI datasets.
"""
import sys
sys.path.insert(0, '.')
from nski.data import download_advbench, download_alpaca, download_harmbench

print('       Starting downloads...')

try:
    path = download_advbench('./data_cache')
    print(f'       + AdvBench: {path}')
except Exception as e:
    print(f'       - AdvBench failed: {e}')

try:
    path = download_alpaca('./data_cache')
    print(f'       + Alpaca: {path}')
except Exception as e:
    print(f'       - Alpaca failed: {e}')

try:
    path = download_harmbench('./data_cache')
    print(f'       + HarmBench: {path}')
except Exception as e:
    print(f'       - HarmBench download note: {e}')

print('       Downloads complete!')
