import random
import logging
import numpy as np

def setup_random_seed(seed=42):
    """
    Set the seed for Python's random module and numpy's random generator to ensure reproducibility.
    """
    seed = random.randint(0, 1000) if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    logging.debug(f"Random seed set to: {seed}")

def setup_logging(log_level=logging.INFO):
    """
    Configure basic logging settings.
    """
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
