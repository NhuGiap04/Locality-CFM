"""
TensorFlow implementation of Conditional Flow Matching.
Adapted from torchcfm for TPU compatibility.

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)
"""

import tensorflow as tf
import numpy as np
from typing import Union, Optional, Tuple


def pad_t_like_x(t, x):
    """
    Reshape the time vector t by the number of dimensions of x.
    
    Parameters
    ----------
    x : tf.Tensor, shape (bs, *dim)
        represents the source minibatch
    t : tf.Tensor, shape (bs,)
    
    Returns
    -------
    tf.Tensor, shape (bs, 1, 1, 1) for 4D x
    """
    if isinstance(t, (float, int)):
        return t
    # Add dimensions to match x (assuming NHWC format for images)
    ndims = len(x.shape)
    for _ in range(ndims - 1):
        t = tf.expand_dims(t, axis=-1)
    return t


class ConditionalFlowMatcher:
    """
    Base class for conditional flow matching methods.
    
    Implements the independent conditional flow matching methods.
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma)
    - conditional flow matching ut(x1|x0) = x1 - x0
    """
    
    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the ConditionalFlowMatcher class.
        
        Parameters
        ----------
        sigma : Union[float, int]
            Standard deviation of the probability path.
        """
        self.sigma = sigma
    
    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma).
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,)
        
        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0
    
    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path.
        
        Parameters
        ----------
        t : tf.Tensor, shape (bs,)
        
        Returns
        -------
        standard deviation sigma
        """
        del t
        return self.sigma
    
    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma).
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,)
        epsilon : tf.Tensor, shape (bs, *dim)
            noise sample from N(0, 1)
        
        Returns
        -------
        xt : tf.Tensor, shape (bs, *dim)
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0) if not isinstance(sigma_t, (float, int)) else sigma_t
        return mu_t + sigma_t * epsilon
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,)
        xt : tf.Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        
        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0
        """
        del t, xt
        return x1 - x0
    
    def sample_noise_like(self, x):
        """Sample Gaussian noise with the same shape as x."""
        return tf.random.normal(tf.shape(x), dtype=x.dtype)
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt and the conditional vector field ut(x1|x0) = x1 - x0.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,), optional
            represents the time levels. If None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon
        
        Returns
        -------
        t : tf.Tensor, shape (bs,)
        xt : tf.Tensor, shape (bs, *dim)
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: tf.Tensor, shape (bs, *dim)
        """
        batch_size = tf.shape(x0)[0]
        
        if t is None:
            t = tf.random.uniform([batch_size], dtype=x0.dtype)
        
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Lipman et al. 2023 style target OT conditional flow matching.
    """
    
    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path tx1.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch (unused)
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,)
        
        Returns
        -------
        mean mu_t: t * x1
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1
    
    def compute_sigma_t(self, t):
        """
        Compute the standard deviation: 1 - (1 - sigma_min) * t.
        """
        return 1 - (1 - self.sigma) * t
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field.
        """
        del x0
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x1)
        return (x1 - (1 - self.sigma) * xt) / sigma_t


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Variance Preserving Conditional Flow Matcher (Stochastic Interpolant).
    """
    
    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path.
        Uses cos/sin interpolation for variance preservation.
        """
        import math
        t = pad_t_like_x(t, x0)
        return tf.cos(math.pi / 2 * t) * x0 + tf.sin(math.pi / 2 * t) * x1
    
    def compute_sigma_t(self, t):
        """Returns the sigma value."""
        del t
        return self.sigma
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field for VP path.
        """
        import math
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (tf.cos(math.pi / 2 * t) * x1 - tf.sin(math.pi / 2 * t) * x0)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Optimal transport conditional flow matching method.
    Uses exact OT plan to pair source and target samples.
    """
    
    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the OT-CFM class.
        
        Parameters
        ----------
        sigma : Union[float, int]
            Standard deviation of the probability path.
        """
        super().__init__(sigma)
        from .optimal_transport import OTPlanSampler
        self.ot_sampler = OTPlanSampler(method="exact")
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt and the conditional vector field with OT pairing.
        
        Parameters
        ----------
        x0 : tf.Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : tf.Tensor, shape (bs, *dim)
            represents the target minibatch
        t : tf.Tensor, shape (bs,), optional
        return_noise : bool
        
        Returns
        -------
        t, xt, ut, (optionally eps)
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
    
    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        """
        Compute the sample xt and conditional flow with labels.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


def get_flow_matcher(model_type: str, sigma: float = 0.0):
    """
    Factory function to create a flow matcher based on the model type.
    
    Parameters
    ----------
    model_type : str
        One of ['otcfm', 'icfm', 'fm', 'si']
    sigma : float
        Standard deviation parameter
    
    Returns
    -------
    ConditionalFlowMatcher
        The appropriate flow matcher instance.
    """
    if model_type == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif model_type == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif model_type == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif model_type == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {model_type}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )
