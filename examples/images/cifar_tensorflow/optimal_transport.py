"""
TensorFlow implementation of Optimal Transport utilities.
Adapted from torchcfm for TPU compatibility.

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)
"""

import numpy as np
import tensorflow as tf
from functools import partial
from typing import Optional, Union
import warnings

# Import POT (Python Optimal Transport) - required
try:
    import ot as pot
except ImportError:
    raise ImportError(
        "Please install POT (Python Optimal Transport): pip install POT"
    )


class OTPlanSampler:
    """
    OTPlanSampler implements sampling coordinates according to an OT plan
    (wrt squared Euclidean cost) with different implementations of the plan calculation.
    """
    
    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """
        Initialize the OTPlanSampler class.
        
        Parameters
        ----------
        method : str
            Choose which optimal transport solver to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced", "partial"]
        reg : float, optional
            Regularization parameter for Sinkhorn-based iterative solvers.
        reg_m : float, optional
            Regularization weight for unbalanced Sinkhorn-Knopp solver.
        normalize_cost : bool, optional
            Normalizes the cost matrix so that the maximum cost is 1.
        num_threads : int or str, optional
            Number of threads for the "exact" OT solver.
        warn : bool, optional
            If True, raises a warning if the algorithm does not converge.
        """
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
    
    def get_map(self, x0, x1):
        """
        Compute the OT plan (wrt squared Euclidean cost) between source and target.
        
        Parameters
        ----------
        x0 : np.ndarray, shape (bs, *dim)
            represents the source minibatch (must be numpy array)
        x1 : np.ndarray, shape (bs, *dim)
            represents the target minibatch (must be numpy array)
        
        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        x0_np = np.asarray(x0)
        x1_np = np.asarray(x1)
        
        a, b = pot.unif(x0_np.shape[0]), pot.unif(x1_np.shape[0])
        
        # Flatten if needed
        if x0_np.ndim > 2:
            x0_flat = x0_np.reshape(x0_np.shape[0], -1)
        else:
            x0_flat = x0_np
        
        if x1_np.ndim > 2:
            x1_flat = x1_np.reshape(x1_np.shape[0], -1)
        else:
            x1_flat = x1_np
        
        # Compute cost matrix (squared Euclidean distance)
        M = np.sum((x0_flat[:, None, :] - x1_flat[None, :, :]) ** 2, axis=-1)
        
        if self.normalize_cost:
            M = M / M.max()
        
        # Compute OT plan
        p = self.ot_fn(a, b, M)
        
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
        
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        
        return p
    
    def sample_map(self, pi, batch_size, replace=True):
        """
        Draw source and target samples from pi (x,z) ~ π
        
        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the OT plan
        batch_size : int
        replace : bool
            sampling with or without replacement
        
        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays
            indices of source and target data samples from π
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])
    
    def sample_plan(self, x0, x1, replace=True):
        """
        Compute the OT plan π and draw source and target samples from pi (x,z) ~ π
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        replace : bool
        
        Returns
        -------
        x0[i], x1[j] : tf.Tensor
            paired samples according to OT plan
        """
        def _sample_indices(x0_np, x1_np):
            """Compute OT indices in eager mode (numpy)."""
            pi = self.get_map(x0_np, x1_np)
            batch_size = x0_np.shape[0]
            i, j = self.sample_map(pi, batch_size, replace=replace)
            return i.astype(np.int32), j.astype(np.int32)
        
        # Use tf.py_function to run numpy operations in eager mode
        i, j = tf.py_function(
            _sample_indices,
            [x0, x1],
            [tf.int32, tf.int32]
        )
        
        # Set shapes for the indices (they will have shape [batch_size])
        batch_size = x0.shape[0] if x0.shape[0] is not None else None
        i.set_shape([batch_size])
        j.set_shape([batch_size])
        
        return tf.gather(x0, i), tf.gather(x1, j)
    
    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        """
        Compute the OT plan π and draw paired samples with labels.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
        x1 : tf.Tensor, shape (bs, *dim)
        y0 : tf.Tensor, shape (bs,), optional
        y1 : tf.Tensor, shape (bs,), optional
        replace : bool
        
        Returns
        -------
        x0[i], x1[j], y0[i], y1[j] : tf.Tensor
        """
        def _sample_indices(x0_np, x1_np):
            """Compute OT indices in eager mode (numpy)."""
            pi = self.get_map(x0_np, x1_np)
            batch_size = x0_np.shape[0]
            i, j = self.sample_map(pi, batch_size, replace=replace)
            return i.astype(np.int32), j.astype(np.int32)
        
        # Use tf.py_function to run numpy operations in eager mode
        i, j = tf.py_function(
            _sample_indices,
            [x0, x1],
            [tf.int32, tf.int32]
        )
        
        # Set shapes for the indices (they will have shape [batch_size])
        batch_size = x0.shape[0] if x0.shape[0] is not None else None
        i.set_shape([batch_size])
        j.set_shape([batch_size])
        
        x0_paired = tf.gather(x0, i)
        x1_paired = tf.gather(x1, j)
        
        if y0 is not None:
            y0_paired = tf.gather(y0, i)
        else:
            y0_paired = None
        
        if y1 is not None:
            y1_paired = tf.gather(y1, j)
        else:
            y1_paired = None
        
        return x0_paired, x1_paired, y0_paired, y1_paired


@tf.function
def compute_pairwise_distances(x0, x1):
    """
    Compute pairwise squared Euclidean distances between two batches.
    TensorFlow-native implementation for TPU compatibility.
    
    Parameters
    ----------
    x0 : tf.Tensor, shape (n, d)
    x1 : tf.Tensor, shape (m, d)
    
    Returns
    -------
    tf.Tensor, shape (n, m)
        Pairwise squared distances
    """
    # x0: [n, d], x1: [m, d]
    # ||x0 - x1||^2 = ||x0||^2 + ||x1||^2 - 2 * x0 @ x1.T
    x0_sq = tf.reduce_sum(x0 ** 2, axis=-1, keepdims=True)  # [n, 1]
    x1_sq = tf.reduce_sum(x1 ** 2, axis=-1, keepdims=True)  # [m, 1]
    cross = tf.matmul(x0, x1, transpose_b=True)  # [n, m]
    distances = x0_sq + tf.transpose(x1_sq) - 2 * cross
    return tf.maximum(distances, 0.0)  # Numerical stability


class TPUFriendlyOTPlanSampler:
    """
    A TPU-friendly OT plan sampler using Sinkhorn algorithm implemented in TensorFlow.
    This avoids the need to transfer data to CPU for POT computation.
    """
    
    def __init__(
        self,
        reg: float = 0.05,
        num_iters: int = 100,
        threshold: float = 1e-9,
    ):
        """
        Initialize the TPU-friendly OT sampler.
        
        Parameters
        ----------
        reg : float
            Entropic regularization parameter.
        num_iters : int
            Maximum number of Sinkhorn iterations.
        threshold : float
            Convergence threshold.
        """
        self.reg = reg
        self.num_iters = num_iters
        self.threshold = threshold
    
    @tf.function
    def sinkhorn(self, M, a=None, b=None):
        """
        Sinkhorn algorithm for computing regularized OT plan.
        
        Parameters
        ----------
        M : tf.Tensor, shape (n, m)
            Cost matrix
        a : tf.Tensor, shape (n,), optional
            Source distribution (uniform if None)
        b : tf.Tensor, shape (m,), optional
            Target distribution (uniform if None)
        
        Returns
        -------
        tf.Tensor, shape (n, m)
            OT plan
        """
        n, m = tf.shape(M)[0], tf.shape(M)[1]
        
        if a is None:
            a = tf.ones([n], dtype=M.dtype) / tf.cast(n, M.dtype)
        if b is None:
            b = tf.ones([m], dtype=M.dtype) / tf.cast(m, M.dtype)
        
        # Kernel
        K = tf.exp(-M / self.reg)
        
        # Initialize
        u = tf.ones([n], dtype=M.dtype)
        v = tf.ones([m], dtype=M.dtype)
        
        # Sinkhorn iterations
        for _ in range(self.num_iters):
            u_prev = u
            u = a / (tf.linalg.matvec(K, v) + 1e-10)
            v = b / (tf.linalg.matvec(K, u, transpose_a=True) + 1e-10)
            
            # Check convergence
            if tf.reduce_max(tf.abs(u - u_prev)) < self.threshold:
                break
        
        # Compute transport plan
        P = tf.expand_dims(u, 1) * K * tf.expand_dims(v, 0)
        return P
    
    def sample_plan(self, x0, x1):
        """
        Compute OT plan and sample paired indices.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
        x1 : tf.Tensor, shape (bs, *dim)
        
        Returns
        -------
        x0_paired, x1_paired : tf.Tensor
        """
        batch_size = tf.shape(x0)[0]
        
        # Flatten for distance computation
        x0_flat = tf.reshape(x0, [batch_size, -1])
        x1_flat = tf.reshape(x1, [batch_size, -1])
        
        # Compute cost matrix
        M = compute_pairwise_distances(x0_flat, x1_flat)
        
        # Compute OT plan
        P = self.sinkhorn(M)
        
        # Sample from the plan (using Gumbel-top-k trick for differentiability)
        # For simplicity, we use argmax per row (deterministic assignment)
        indices = tf.argmax(P, axis=1, output_type=tf.int32)
        
        x1_paired = tf.gather(x1, indices)
        
        return x0, x1_paired
