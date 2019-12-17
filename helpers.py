# helpers.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-12-14
# IFT-6390

from scikit_classifiers import DATASETS
from scikit_classifiers import CLASSIFIERS

def print_models():
    _MODELS = list(CLASSIFIERS.keys())
    print(f"Models")
    for i, m in enumerate(_MODELS):
        print(f"{i}: {m}")

def print_datasets():
    _DATASETS = list(DATASETS.keys())
    print(f"Datasets")
    for i, m in enumerate(_DATASETS):
        print(f"{i}: {m}")
