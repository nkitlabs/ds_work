import tensorflow as tf
import numpy as np

GELU_MULTIPLYER_CONST = np.sqrt(2/np.pi)

def gelu_activate_fn(x):
    """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
        x: float Tensor to perform activation.
        Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(GELU_MULTIPLYER_CONST * x * (1+0.044715*x*x)))
    return x * cdf 