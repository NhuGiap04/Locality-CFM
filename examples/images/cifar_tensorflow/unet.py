"""
TensorFlow UNet implementation for Conditional Flow Matching.
Adapted from the PyTorch version for TPU compatibility.

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Parameters
    ----------
    timesteps : tf.Tensor
        a 1-D Tensor of N indices, one per batch element.
    dim : int
        the dimension of the output.
    max_period : int
        controls the minimum frequency of the embeddings.
    
    Returns
    -------
    tf.Tensor
        an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = tf.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.cast(timesteps, tf.float32)[:, None] * freqs[None, :]
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class GroupNormalization(layers.Layer):
    """Group Normalization layer."""
    
    def __init__(self, groups=32, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(channels,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = x.shape[-1]
        
        # Reshape for group normalization
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels // self.groups])
        
        # Compute mean and variance
        mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        
        # Reshape back
        x = tf.reshape(x, [batch_size, height, width, channels])
        
        # Apply scale and shift
        return x * self.gamma + self.beta


class ResBlock(layers.Layer):
    """
    A residual block that can optionally change the number of channels.
    """
    
    def __init__(self, channels, emb_channels, dropout, out_channels=None, 
                 use_scale_shift_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dropout_rate = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Input layers
        self.in_norm = GroupNormalization(groups=32)
        self.in_conv = layers.Conv2D(
            self.out_channels, 3, padding='same',
            kernel_initializer='he_normal'
        )
        
        # Embedding projection
        self.emb_proj = layers.Dense(
            self.out_channels * 2 if use_scale_shift_norm else self.out_channels
        )
        
        # Output layers
        self.out_norm = GroupNormalization(groups=32)
        self.out_conv = layers.Conv2D(
            self.out_channels, 3, padding='same',
            kernel_initializer='zeros'
        )
        self.dropout = layers.Dropout(dropout)
        
        # Skip connection
        if channels != self.out_channels:
            self.skip_conv = layers.Conv2D(
                self.out_channels, 1, padding='same',
                kernel_initializer='he_normal'
            )
        else:
            self.skip_conv = None
    
    def call(self, x, emb, training=None):
        h = x
        h = self.in_norm(h)
        h = tf.nn.silu(h)
        h = self.in_conv(h)
        
        # Add timestep embedding
        emb_out = self.emb_proj(tf.nn.silu(emb))[:, None, None, :]
        
        if self.use_scale_shift_norm:
            scale, shift = tf.split(emb_out, 2, axis=-1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = h + emb_out
            h = self.out_norm(h)
        
        h = tf.nn.silu(h)
        h = self.dropout(h, training=training)
        h = self.out_conv(h)
        
        # Skip connection
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        
        return x + h


class AttentionBlock(layers.Layer):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    
    def __init__(self, channels, num_heads=1, num_head_channels=-1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels
        
        self.norm = GroupNormalization(groups=32)
        self.qkv = layers.Conv2D(channels * 3, 1, padding='same')
        self.proj_out = layers.Conv2D(channels, 1, padding='same', kernel_initializer='zeros')
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = x.shape[-1]
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for attention
        qkv = tf.reshape(qkv, [batch_size, height * width, 3, self.num_heads, channels // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, num_heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(channels // self.num_heads)
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)
        
        h = tf.matmul(attn, v)  # [B, num_heads, HW, head_dim]
        h = tf.transpose(h, [0, 2, 1, 3])  # [B, HW, num_heads, head_dim]
        h = tf.reshape(h, [batch_size, height, width, channels])
        
        h = self.proj_out(h)
        return x + h


class Downsample(layers.Layer):
    """A downsampling layer with strided convolution."""
    
    def __init__(self, channels, use_conv=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        if use_conv:
            self.op = layers.Conv2D(channels, 3, strides=2, padding='same')
        else:
            self.op = layers.AveragePooling2D(pool_size=2, strides=2)
    
    def call(self, x):
        return self.op(x)


class Upsample(layers.Layer):
    """An upsampling layer with optional convolution."""
    
    def __init__(self, channels, use_conv=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = layers.Conv2D(channels, 3, padding='same')
    
    def call(self, x):
        x = tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[2] * 2], method='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class UNetModel(keras.Model):
    """
    The full UNet model with attention and timestep embedding.
    
    Parameters
    ----------
    image_size : int
        Size of the input image (assumes square images).
    in_channels : int
        Number of input channels.
    model_channels : int
        Base channel count for the model.
    out_channels : int
        Number of output channels.
    num_res_blocks : int
        Number of residual blocks per downsample.
    attention_resolutions : tuple
        A collection of downsample rates at which attention will take place.
    dropout : float
        The dropout probability.
    channel_mult : tuple
        Channel multiplier for each level of the UNet.
    num_heads : int
        The number of attention heads in each attention layer.
    num_head_channels : int
        If specified, ignore num_heads and instead use a fixed channel width per attention head.
    """
    
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        
        time_embed_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = keras.Sequential([
            layers.Dense(time_embed_dim, activation='silu'),
            layers.Dense(time_embed_dim),
        ])
        
        # Input convolution
        ch = model_channels
        self.input_conv = layers.Conv2D(ch, 3, padding='same')
        
        # Downsampling blocks
        self.input_blocks = []
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block_ch = int(mult * model_channels)
                block_layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=block_ch,
                            use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = block_ch
                if ds in attention_resolutions:
                    block_layers.append(
                        AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
                    )
                self.input_blocks.append(block_layers)
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append([Downsample(ch)])
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = [
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
        ]
        
        # Upsampling blocks
        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                block_ch = int(model_channels * mult)
                block_layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=block_ch,
                            use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = block_ch
                if ds in attention_resolutions:
                    block_layers.append(
                        AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
                    )
                if level and i == num_res_blocks:
                    block_layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(block_layers)
        
        # Output layers
        self.out_norm = GroupNormalization(groups=32)
        self.out_conv = layers.Conv2D(out_channels, 3, padding='same', kernel_initializer='zeros')
    
    def call(self, inputs, training=None):
        """
        Apply the model to an input batch.
        
        Parameters
        ----------
        inputs : tuple
            (timesteps, x) where timesteps is [N] and x is [N, H, W, C]
        
        Returns
        -------
        tf.Tensor
            [N, H, W, C] tensor of outputs.
        """
        t, x = inputs
        
        # Ensure t is 1D
        if len(t.shape) > 1:
            t = t[:, 0]
        if len(t.shape) == 0:
            t = tf.fill([tf.shape(x)[0]], t)
        
        # Time embedding
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        
        # Input
        h = self.input_conv(x)
        hs = [h]
        
        # Downsampling
        for block_layers in self.input_blocks:
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb, training=training)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb, training=training)
            else:
                h = layer(h)
        
        # Upsampling
        for block_layers in self.output_blocks:
            h = tf.concat([h, hs.pop()], axis=-1)
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb, training=training)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = tf.nn.silu(h)
        h = self.out_conv(h)
        
        return h


class UNetModelWrapper(keras.Model):
    """
    Wrapper for UNet model with simplified interface matching PyTorch version.
    """
    
    def __init__(
        self,
        dim,
        num_channels,
        num_res_blocks,
        channel_mult=None,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        dropout=0,
        use_scale_shift_norm=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        dim : tuple
            (C, H, W) input dimensions
        """
        super().__init__(**kwargs)
        
        image_size = dim[-1]
        in_channels = dim[0]
        
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (1, 2, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(channel_mult)
        
        # Parse attention resolutions
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
        )
    
    def call(self, t, x, training=None):
        """
        Forward pass matching PyTorch interface: model(t, x)
        
        Note: TensorFlow uses NHWC format, so we expect x to be [N, H, W, C]
        """
        return self.unet((t, x), training=training)


def create_unet_cifar10(num_channels=128, dropout=0.1):
    """
    Create a UNet model configured for CIFAR-10.
    
    Parameters
    ----------
    num_channels : int
        Base number of channels.
    dropout : float
        Dropout rate.
    
    Returns
    -------
    UNetModelWrapper
        The configured model.
    """
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=dropout,
    )
