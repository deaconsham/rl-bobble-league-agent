"""
shared test setup. puts bobble_model/Scripts and tests on the import path, and seeds
python, numpy and torch before every test so T1 and T2 repeat exactly.
"""
import os
import random
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, 'bobble_model', 'Scripts')
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

for p in (MODEL_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def seed_everything():
    """
    Docstring for seed_everything:

    seed before each test. torch is optional for the pure-protocol tests.
    """
    random.seed(0)
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except Exception:
        pass
    yield
