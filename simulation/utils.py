# utils.py
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
