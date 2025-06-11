import math
import numpy as np

from functools import partial
from typing import Any, Union, Sequence, Optional, Tuple

from jax.experimental import host_callback as hcb

import sys
import jax
import jax.numpy as jnp
import flax
from jax import vmap
from jax.tree_util import tree_map
from jax.sharding import PartitionSpec as PS
from jax import debug
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from ttt.infra.jax_utils import with_sharding_constraint, get_gradient_checkpoint_policy

Axes = Union[int, Sequence[int]]


def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """
    Remat every n mini batches.
    """
    x_grouped = tree_map(lambda x: x.reshape((-1, n, *x.shape[1:])), x)
    carry, y_grouped = jax.lax.scan(jax.remat(partial(jax.lax.scan, f), prevent_cse=False), carry, x_grouped)
    y = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), y_grouped)
    return carry, y


def get_multi_head_params(self, params, param_dtype, kernel_init="normal", std=0.02):
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")
    for k in flat_params.keys():
        new_shape = (self.num_heads, *flat_params[k].shape)
        if "scale" in k:
            p = self.param(k, jax.nn.initializers.ones, new_shape, param_dtype)
        elif "kernel" in k:
            if kernel_init == "normal":
                initializer = nn.initializers.normal(std)
            elif kernel_init == "zeros":
                initializer = nn.initializers.zeros
            elif kernel_init == "ones":
                initializer = nn.initializers.ones
            else:
                raise NotImplementedError("Initializer %s Not Implemented." % (kernel_init))
            p = self.param(k, initializer, new_shape, param_dtype)
        else:
            p = self.param(k, jax.nn.initializers.zeros, new_shape, param_dtype)
        flat_params[k] = p
    params_init = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    return params_init


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)
    freqs = np.outer(t, freqs).astype(dtype)
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray, dtype: jnp.dtype = jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


def diff_gelu(x):
    tanh_out = jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class LinearLayerTemplate(nn.Module):
    width: int
    use_bias: bool
    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.width, use_bias=self.use_bias, name=self.name, dtype=self.dtype, param_dtype=self.param_dtype
        )(x)
        return x


class LayerNormTemplate(nn.Module):
    name: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(name=self.name, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class TTTBase(nn.Module):
    config: Any = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.width = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = self.config.mini_batch_size
        self.n_mini_batch = self.config.max_sequence_length // self.mini_batch_size
        self.seq_shape = (self.n_mini_batch, self.mini_batch_size)
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, self.mini_batch_size * 2, theta=self.config.rope_theta, dtype=self.dtype
        )

        self.setup_qkvo()
        self.setup_token_idx()
        self.setup_ttt_lr_gate()

        self.ttt_norm = LayerNormTemplate(dtype=self.dtype, param_dtype=self.param_dtype)
        ttt_norm_params = self.ttt_norm.init(jax.random.PRNGKey(0), jnp.ones([1, self.head_dim]))["params"]
        self.ttt_norm_params = get_multi_head_params(
            self, ttt_norm_params, param_dtype=self.param_dtype, kernel_init="layer_norm"
        )
        self.post_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)

        self.ttt_params = ()

        # Initialize caches - TTT cache for parameters and conv cache for convolutional states
        if self.config.use_cache:
            print("Initializing TTT and conv caches.")
            self.ttt_cache = self.variable('ttt_cache', 'weights', lambda: ())
            self.conv_cache = self.variable('conv_cache', 'states', lambda: ())

    def _init_conv_cache_if_needed(self, batch_size: int):
        """Initialize conv cache if it doesn't exist or is empty. Efficient implementation."""
        if self.is_mutable_collection('conv_cache') and self.conv_cache.value == ():
            if hasattr(self, 'conv_q') and hasattr(self, 'conv_k'):
                conv_kernel_size = getattr(self.config, 'conv_width', 4)
                
                # Cache stores the last (K-1) input elements for each conv layer
                # This is all we need for efficient causal convolution
                cache_size = max(0, conv_kernel_size - 1)
                
                conv_states_dict = {
                    'conv_q_state': jnp.zeros((batch_size, cache_size, self.width), dtype=self.dtype),
                    'conv_k_state': jnp.zeros((batch_size, cache_size, self.width), dtype=self.dtype),
                }
                self.conv_cache.value = conv_states_dict
            
    def _get_conv_cache_or_none(self, batch_size: int):
        """Get conv cache dictionary, initializing if needed."""
        if not self.is_mutable_collection('conv_cache'):
            return None
            
        if self.conv_cache.value == ():
            self._init_conv_cache_if_needed(batch_size)
            
        return self.conv_cache.value if isinstance(self.conv_cache.value, dict) else None

    def _apply_causal_conv_with_cache(self, x, conv_layer, cache_key, batch_size):
        """
        Apply causal convolution with proper caching for efficiency.
        
        For a 1D causal conv with kernel size K:
        - Maintains a cache of the last (K-1) input elements  
        - For new input, concatenates [cached_context, new_input] and applies conv
        - Returns only outputs corresponding to new input positions
        - Updates cache with the last (K-1) elements from new input
        
        This provides O(1) computation per new token instead of O(sequence_length).
        """

        use_cache = self.config.use_cache

        cache_dict = self._get_conv_cache_or_none(batch_size) if use_cache else None
        
        if cache_dict is None or not use_cache:
            # No caching available or disabled - fall back to full computation
            return conv_layer(x)
        
        # Get kernel size from config (static value) to avoid traced array issues
        conv_kernel_size = getattr(self.config, 'conv_width', 4)
        cache_size = max(0, conv_kernel_size - 1)
        seq_len = x.shape[1]
        
        # Initialize cache for this layer if needed
        if cache_key not in cache_dict:
            cache_dict[cache_key] = jnp.zeros((batch_size, cache_size, self.width), dtype=x.dtype)
        
        if cache_size == 0:
            # Kernel size 1 - no context needed, compute directly
            conv_output = conv_layer(x)
        else:
            cached_context = cache_dict[cache_key]  # Shape: (B, K-1, C)
            
            # Concatenate cached context with new input
            full_input = jnp.concatenate([cached_context, x], axis=1)  # Shape: (B, K-1+L, C)
            
            # Apply convolution to full sequence
            full_output = conv_layer(full_input)  # Shape: (B, K-1+L, C)
            
            # Extract outputs corresponding only to new input positions
            # Use dynamic slice to avoid traced indexing issues
            conv_output = jax.lax.dynamic_slice(
                full_output,
                (0, full_output.shape[1] - seq_len, 0),
                (batch_size, seq_len, self.width)
            )
        
        # Update cache with the last (K-1) elements from the new input
        if cache_size > 0:
            cached_context = cache_dict[cache_key]
            
            if seq_len >= cache_size:
                # Take the last (K-1) elements from new input using dynamic slice
                new_cache_state = jax.lax.dynamic_slice(
                    x,
                    (0, seq_len - cache_size, 0),
                    (batch_size, cache_size, self.width)
                )
            else:
                # New input is shorter than cache size
                # Combine old cache + new input, then take last (K-1) elements
                combined = jnp.concatenate([cached_context, x], axis=1)
                combined_len = combined.shape[1]
                new_cache_state = jax.lax.dynamic_slice(
                    combined,
                    (0, combined_len - cache_size, 0), 
                    (batch_size, cache_size, self.width)
                )
            
            cache_dict[cache_key] = new_cache_state
        
        # Update sequence offset
        self.conv_cache.value = cache_dict  # Update the mutable variable
        
        return conv_output

    def reset_conv_cache(self):
        """Reset conv cache for new sequences."""
        if self.is_mutable_collection('conv_cache') and isinstance(self.conv_cache.value, dict):
            cache_dict = self.conv_cache.value
            
            # Zero out cached states
            if 'conv_q_state' in cache_dict:
                cache_dict['conv_q_state'] = jnp.zeros_like(cache_dict['conv_q_state'])
            if 'conv_k_state' in cache_dict:
                cache_dict['conv_k_state'] = jnp.zeros_like(cache_dict['conv_k_state'])

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        position_ids=None,
        deterministic: bool = True,
        output_ttt_stats: bool = False,
        ttt_lr_mult=1.0,
        reset_cache: bool = False
    ):
        if reset_cache:
            self.reset_conv_cache()
            if self.is_mutable_collection('ttt_cache'):
                self.ttt_cache.value = ()
        
        B = hidden_states.shape[0]
        # Initialize TTT parameter cache if needed

        # check if has attribute `ttt_cache` and if it is mutable
        if self.is_mutable_collection('ttt_cache') and hasattr(self, 'ttt_cache') and  self.ttt_cache.value == ():
            print("Initializing TTT parameter cache.")
            
            batched_ttt_params = tree_map(lambda p: p[None].repeat(B, axis=0) if isinstance(p, jnp.ndarray) else p, self.ttt_params)
            self.ttt_cache.value = batched_ttt_params

        self.config.output_ttt_stats = output_ttt_stats
        del deterministic
        
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(hidden_states, position_ids=position_ids)
        eta *= ttt_lr_mult

        Z, ttt_stats = self.ttt(XQ, XK, XV, eta)
        
        Z = self.post_norm(Z)
        Z = self.apply_gate(hidden_states, Z)
        ttt_output = self.project_ttt_outputs(Z)

        _ttt_loss_mse_init, _ttt_loss_mse_step_0, _ttt_loss_mse_step_1, ttt_params_final = ttt_stats

        # Update TTT cache with the final parameters from the scan
        if self.is_mutable_collection('ttt_cache') and hasattr(self, 'ttt_cache') :
            print("Updating TTT parameter cache with final parameters.", len(ttt_params_final))
            self.ttt_cache.value = ttt_params_final
        
        return ttt_output, (
             precompute_stats, 
             _ttt_loss_mse_init, 
             _ttt_loss_mse_step_0, 
             _ttt_loss_mse_step_1
        )

    def setup_qkvo(self):
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_token_idx(self):
        self.token_idx = 1.0 / jnp.arange(1, self.mini_batch_size + 1, dtype=jnp.float32)
        self.learnable_token_idx = self.param(
            "learnable_token_idx", nn.initializers.zeros, (self.mini_batch_size,), jnp.float32
        )

    def setup_ttt_lr_gate(self):
        self.learnable_ttt_lr = LinearLayerTemplate(
            width=1, use_bias=True, name="learnable_ttt_lr", dtype=self.dtype, param_dtype=self.param_dtype
        )
        learnable_ttt_lr_params = self.learnable_ttt_lr.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
        self.learnable_ttt_lr_params = get_multi_head_params(
            self,
            learnable_ttt_lr_params,
            param_dtype=self.param_dtype,
            kernel_init="normal",
            std=self.config.initializer_range,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_mini_batches(self, hidden_states):
        B, N, num_head, head_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(B, *self.seq_shape, self.num_heads, self.head_dim).transpose(
            0, 3, 1, 2, 4
        )
        return hidden_states

    def get_qkv_projections(self, batch):
        """Default implementation - can be overridden in subclasses for conv caching."""
        XQ, XK, XV = self.wq(batch), self.wk(batch), self.wv(batch)
        return XQ, XK, XV

    def get_eta(self, X):
        learnable_ttt_lr = vmap(
            lambda x, p: self.learnable_ttt_lr.apply({"params": p}, x), axis_name="head", in_axes=[None, 0], out_axes=1
        )(X, self.learnable_ttt_lr_params)
        learnable_ttt_lr = nn.sigmoid(learnable_ttt_lr)
        learnable_ttt_lr = learnable_ttt_lr.transpose(0, 1, 2, 4, 3)

        token_idx = self.learnable_token_idx + self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        eta = (
            (self.config.ttt_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            / self.head_dim
        )
        return eta

    def get_ttt_inputs(self, batch, position_ids):
        B, N, F = batch.shape
        n_mini_batch = N // self.mini_batch_size
        X = batch.reshape(B, *self.seq_shape, self.width)

        XQ, XK, XV = self.get_qkv_projections(batch)

        if self.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            XK_last_in_mini_batch = XK[:, :: self.mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(B, 1, self.num_heads, self.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = ((ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        XQ = with_sharding_constraint(XQ, PS(("dp", "fsdp"), None, "mp"))
        XK = with_sharding_constraint(XK, PS(("dp", "fsdp"), None, "mp"))
        XV = with_sharding_constraint(XV, PS(("dp", "fsdp"), None, "mp"))

        XQ = self._split_heads(XQ)
        XK = self._split_heads(XK)
        XV = self._split_heads(XV)

        freqs_cis = jnp.take(self.freqs_cis, position_ids % self.mini_batch_size, axis=0)
        XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=self.dtype)

        XQ = self._split_mini_batches(XQ)
        XK = self._split_mini_batches(XK)
        XV = self._split_mini_batches(XV)

        eta = self.get_eta(X)

        return (XQ, XK, XV, eta, ssl_tgt_last_in_mini_batch_from_mean_mse)

    def apply_gate(self, hidden_states, ttt_output):
        return ttt_output

    def project_ttt_outputs(self, XQW_batch):
        z_batch = self.wo(XQW_batch)
        return z_batch

    def ttt(self, XQ, XK, XV, eta):
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta, ttt_params=None):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, ttt_params_init, ttt_norm_params):
                def compute_mini_batch(ttt_params_mini_batch_init, inputs):
                    XQ_mini_batch = inputs["XQ"]
                    XK_mini_batch = inputs["XK"]
                    XV_mini_batch = inputs["XV"]
                    eta_mini_batch = inputs["eta"]

                    ttt_params_last_in_mini_batch, outputs = self.process_mini_batch(
                        XQ_mini_batch,
                        XK_mini_batch,
                        XV_mini_batch,
                        eta_mini_batch,
                        ttt_params_init,
                        ttt_params_mini_batch_init,
                        ttt_norm_params,
                    )
                    return ttt_params_last_in_mini_batch, outputs

                inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}

                ttt_params_final, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch, self.config.remat_mini_batch_group_size, ttt_params_init, inputs
                )
                Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs

                return (Z.reshape(-1, self.head_dim), ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, ttt_params_final)

            outputs = parallelize_over_heads(XQ, XK, XV, eta,  self.ttt_params if  ttt_params is None else ttt_params , self.ttt_norm_params)
            return outputs

        outputs = update_embed(XQ, XK, XV, eta, ttt_params=self.ttt_cache.value if hasattr(self, 'ttt_cache') else None)
        Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, ttt_params_final = outputs

        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, -1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_init = ttt_loss_mse_init.mean(axis=(0, 1))
            ttt_loss_mse_step_0 = ttt_loss_mse_step_0.mean(axis=(0, 1))
            ttt_loss_mse_step_1 = ttt_loss_mse_step_1.mean(axis=(0, 1))

        return Z, (ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, ttt_params_final)


class TTTLinearBase(TTTBase):
    def setup(self):
        super().setup()

        self.W1 = self.param(
            "ttt_dense_0",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, self.head_dim, self.head_dim),
            self.param_dtype,
        )
        
        self.ttt_params = (self.W1,)

    def process_mini_batch(
            self,
            XQ_mini_batch,
            XK_mini_batch,
            XV_mini_batch,
            eta_mini_batch,
            ttt_params_init,
            ttt_params_mini_batch_init,
            ttt_norm_params
        ):
            W1_init, = ttt_params_mini_batch_init
            square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
            last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init
            ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z1)
            ssl_target = XV_mini_batch - XK_mini_batch
            grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
            grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]

            # Calculate TTT loss using W_init of the current mini-batch
            if self.config.output_ttt_stats:
                ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
            else:
                ttt_loss_mse_step_0 = None

            # Calculate TTT loss using W_init of the entire sequence
            if self.config.output_ttt_stats:
                W1_0, = ttt_params_init
                Z1_0 = X1 @ W1_0
                ttt_norm_out_0 = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_0)
                ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
            else:
                ttt_loss_mse_init = None

            # Original adaptive behavior
            X1_bar = XQ_mini_batch
            Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))
            Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1
            ttt_norm_out_bar = self.ttt_norm.apply({"params": ttt_norm_params}, Z1_bar)

            output_mini_batch = X1_bar + ttt_norm_out_bar

            W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1

            # Calculate ttt loss using the updated W_init by the current mini-batch
            if self.config.output_ttt_stats:
                X1_last_fwd_new = X1[-1:] @ W1_bar_last
                X1_last_fwd_new = self.ttt_norm.apply({"params": ttt_norm_params}, X1_last_fwd_new)
                ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
            else:
                ttt_loss_mse_step_1 = None

            ttt_params_mini_batch_new = (W1_bar_last,)

            return (
                ttt_params_mini_batch_new,
                (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
            )


class TTTLinear(TTTLinearBase):
    def setup(self):
        super().setup()
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        # Shared Q/K projection
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        
        # Separate convolutions for Q and K
        if self.config.remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(self.config.remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
            
        self.conv_q = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        
        # V projection
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        
        # Output projection
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def get_qkv_projections(self, batch):
        """Get Q, K, V projections with efficient conv caching for shared Q/K."""

        batch_size = batch.shape[0]
        xqk, XV = self.wq(batch), self.wv(batch)
        
        # Apply convolutions with efficient caching
        XQ = self._apply_causal_conv_with_cache(xqk, self.conv_q, 'conv_q_state', batch_size)
        XK = self._apply_causal_conv_with_cache(xqk, self.conv_k, 'conv_k_state', batch_size)
        
        return XQ, XK, XV

    def apply_gate(self, hidden_states, ttt_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output


class TTTMLPBase(TTTBase):
    def setup(self):
        super().setup()
        self.W1 = self.param(
            "ttt_dense_0",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, self.head_dim, 4 * self.head_dim),
            self.param_dtype,
        )
        self.W2 = self.param(
            "ttt_dense_1",
            nn.initializers.normal(self.config.initializer_range),
            (self.num_heads, 4 * self.head_dim, self.head_dim),
            self.param_dtype,
        )
        self.ttt_params = (self.W1, self.W2)

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        ttt_params_init,
        ttt_params_mini_batch_init,
        ttt_norm_params
    ):
        W1_init, W2_init = ttt_params_mini_batch_init
        square_eta_mini_batch = eta_mini_batch[: self.mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch
        Z1 = X1 @ W1_init
        X2 = nn.gelu(Z1)
        Z2 = X2 @ W2_init
        ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z2)

        ssl_target = XV_mini_batch - X1
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z2 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]
        grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(1, 0) * diff_gelu(Z1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate ttt loss using W_init of the entire sequence
        if self.config.output_ttt_stats:
            W1_0, W2_0 = ttt_params_init
            Z1_0 = X1 @ W1_0
            X2_0 = nn.gelu(Z1_0)
            Z2_0 = X2_0 @ W2_0
            ttt_norm_out_0 = self.ttt_norm.apply({"params": ttt_norm_params}, Z2_0)
            ttt_loss_mse_init = ((ttt_norm_out_0 - ssl_target)[-1] ** 2).mean()
        else:
            ttt_loss_mse_init = None

        # Original adaptive behavior
        X1_bar = XQ_mini_batch
        Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0))
        Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1

        X2_bar = nn.gelu(Z1_bar)
        Attn2 = jnp.tril(X2_bar @ X2.transpose(1, 0))
        Z2_bar = X2_bar @ W2_init - (square_eta_mini_batch * Attn2) @ grad_l_wrt_Z2
        ttt_norm_out_bar = self.ttt_norm.apply({"params": ttt_norm_params}, Z2_bar)

        output_mini_batch = X1_bar + ttt_norm_out_bar

        W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
        W2_bar_last = W2_init - (last_eta_in_mini_batch * X2).transpose(1, 0) @ grad_l_wrt_Z2

        if self.config.output_ttt_stats:
            X1_last_fwd_new = nn.gelu(X1[-1:] @ W1_bar_last) @ W2_bar_last
            X1_last_fwd_new = self.ttt_norm.apply({"params": ttt_norm_params}, X1_last_fwd_new)
            ttt_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            ttt_loss_mse_step_1 = None

        ttt_params_mini_batch_new = (W1_bar_last, W2_bar_last)

        return (
            ttt_params_mini_batch_new,
            (output_mini_batch, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1),
        )


class TTTMLP(TTTMLPBase):
    def setup(self):
        super().setup()
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        # Shared Q/K projection
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        
        # Separate convolutions for Q and K with caching support
        if self.config.remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(self.config.remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
            
        self.conv_q = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (self.config.conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        
        # V projection
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        
        # Output projection
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def get_qkv_projections(self, batch):
        """Get Q, K, V projections with efficient conv caching for shared Q/K."""
        if hasattr(self, 'conv_q') and hasattr(self, 'conv_k'):
            batch_size = batch.shape[0]
            xqk, XV = self.wq(batch), self.wv(batch)
            
            # Apply convolutions with efficient caching
            XQ = self._apply_causal_conv_with_cache(xqk, self.conv_q, 'conv_q_state', batch_size)
            XK = self._apply_causal_conv_with_cache(xqk, self.conv_k, 'conv_k_state', batch_size)
            
            return XQ, XK, XV
        else:
            # Standard separate projections (retrocompatible fallback)
            XQ, XK, XV = self.wq(batch), self.wk(batch), self.wv(batch)
            return XQ, XK, XV

    def apply_gate(self, hidden_states, ttt_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output