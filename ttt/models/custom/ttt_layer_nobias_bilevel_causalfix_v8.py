import math
import numpy as np
from functools import partial
from typing import Any, Union, Sequence, Optional, Tuple
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

# Utility functions
Axes = Union[int, Sequence[int]]

def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """Remat every n mini batches."""
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
            elif kernel_init == "layer_norm":
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

def with_sharding_constraint(x, constraint):
    """Placeholder for sharding constraint - implement based on your distributed setup."""
    return x

def get_gradient_checkpoint_policy(policy_name):
    """Placeholder for gradient checkpoint policy."""
    return None

# Template modules
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

# Main TTT implementation
class TTTBase(nn.Module):
    config: Any = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.width = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        
        # Fast path mini-batch size (regular TTT)
        self.mini_batch_size_fast = self.config.mini_batch_size
        
        # Slow path mini-batch size (configurable multiplier, default 2x)
        slow_multiplier = getattr(self.config, 'slow_mini_batch_multiplier', 2)
        self.mini_batch_size_slow = self.mini_batch_size_fast * slow_multiplier
        
        # Ensure sequence length is divisible by both mini-batch sizes
        assert self.config.max_sequence_length % self.mini_batch_size_slow == 0
        
        self.n_mini_batch_fast = self.config.max_sequence_length // self.mini_batch_size_fast
        self.n_mini_batch_slow = self.config.max_sequence_length // self.mini_batch_size_slow
        
        self.seq_shape_fast = (self.n_mini_batch_fast, self.mini_batch_size_fast)
        self.seq_shape_slow = (self.n_mini_batch_slow, self.mini_batch_size_slow)
        
        # RoPE for both paths
        rope_theta = getattr(self.config, 'rope_theta', 10000.0)
        self.freqs_cis_fast = precompute_freqs_cis(
            self.head_dim, self.mini_batch_size_fast * 2, theta=rope_theta, dtype=self.dtype
        )
        self.freqs_cis_slow = precompute_freqs_cis(
            self.head_dim, self.mini_batch_size_slow * 2, theta=rope_theta, dtype=self.dtype
        )

        self.setup_qkvo()
        self.setup_token_idx()
        self.setup_ttt_lr_gate()
        self.setup_mixing_weights()

        # TTT norm shared between paths
        self.ttt_norm = LayerNormTemplate(name="ttt_norm", dtype=self.dtype, param_dtype=self.param_dtype)
        ttt_norm_params = self.ttt_norm.init(jax.random.PRNGKey(0), jnp.ones([1, self.head_dim]))["params"]
        self.ttt_norm_params = get_multi_head_params(
            self, ttt_norm_params, param_dtype=self.param_dtype, kernel_init="layer_norm"
        )
        self.post_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)

        # Separate TTT parameters for fast and slow paths (to be set by subclasses)
        self.ttt_params_fast = ()
        self.ttt_params_slow = ()

        # Initialize caches for both paths
        if getattr(self.config, 'use_cache', False):
            self.ttt_cache_fast = self.variable('ttt_cache_fast', 'weights_fast', lambda: ())
            self.ttt_cache_slow = self.variable('ttt_cache_slow', 'weights_slow', lambda: ())
            self.conv_cache = self.variable('conv_cache', 'states', lambda: ())

    def setup_mixing_weights(self):
        """Setup learnable weights for mixing fast and slow path outputs at logit level."""
        # Learnable mixing coefficient (logit space)
        # Initialize to -2.0 so slow path starts with ~0.12 weight (sigmoid(-2) ≈ 0.12)
        self.mixing_logit = self.param(
            "mixing_logit", 
            nn.initializers.constant(-2.0),
            (), 
            self.param_dtype
        )
        
        # Optional: per-head mixing weights for more flexibility
        self.per_head_mixing = getattr(self.config, 'per_head_mixing', False)
        if self.per_head_mixing:
            self.mixing_logit_per_head = self.param(
                "mixing_logit_per_head",
                nn.initializers.constant(-2.0),
                (self.num_heads,),
                self.param_dtype
            )

    def get_mixing_weight(self, head_idx=None):
        """Get mixing weight(s) for combining fast and slow paths."""
        if self.per_head_mixing and head_idx is not None:
            logit = self.mixing_logit_per_head[head_idx]
        else:
            logit = self.mixing_logit
        
        # Sigmoid to get weight in [0, 1]
        # Fast path weight: 1 - sigmoid(logit)
        # Slow path weight: sigmoid(logit)
        slow_weight = nn.sigmoid(logit)
        fast_weight = 1.0 - slow_weight
        return fast_weight, slow_weight

    def _init_conv_cache_if_needed(self, batch_size: int):
        """Initialize conv cache if it doesn't exist or is empty."""
        if (self.is_mutable_collection('conv_cache') and 
            hasattr(self, 'conv_cache') and 
            self.conv_cache.value == ()):
            if hasattr(self, 'conv_q') and hasattr(self, 'conv_k'):
                conv_kernel_size = getattr(self.config, 'conv_width', 4)
                cache_size = max(0, conv_kernel_size - 1)
                
                conv_states_dict = {
                    'conv_q_state': jnp.zeros((batch_size, cache_size, self.width), dtype=self.dtype),
                    'conv_k_state': jnp.zeros((batch_size, cache_size, self.width), dtype=self.dtype),
                }
                self.conv_cache.value = conv_states_dict
            
    def _get_conv_cache_or_none(self, batch_size: int):
        """Get conv cache dictionary, initializing if needed."""
        if not self.is_mutable_collection('conv_cache') or not hasattr(self, 'conv_cache'):
            return None
            
        if self.conv_cache.value == ():
            self._init_conv_cache_if_needed(batch_size)
            
        return self.conv_cache.value if isinstance(self.conv_cache.value, dict) else None

    def _apply_causal_conv_with_cache(self, x, conv_layer, cache_key, batch_size):
        """Apply causal convolution with proper caching for efficiency."""
        use_cache = getattr(self.config, 'use_cache', False)
        cache_dict = self._get_conv_cache_or_none(batch_size) if use_cache else None
        
        if cache_dict is None or not use_cache:
            return conv_layer(x)
        
        conv_kernel_size = getattr(self.config, 'conv_width', 4)
        cache_size = max(0, conv_kernel_size - 1)
        seq_len = x.shape[1]
        
        if cache_key not in cache_dict:
            cache_dict[cache_key] = jnp.zeros((batch_size, cache_size, self.width), dtype=x.dtype)
        
        if cache_size == 0:
            conv_output = conv_layer(x)
        else:
            cached_context = cache_dict[cache_key]
            full_input = jnp.concatenate([cached_context, x], axis=1)
            full_output = conv_layer(full_input)
            
            conv_output = jax.lax.dynamic_slice(
                full_output,
                (0, full_output.shape[1] - seq_len, 0),
                (batch_size, seq_len, self.width)
            )
        
        if cache_size > 0:
            if seq_len >= cache_size:
                new_cache_state = jax.lax.dynamic_slice(
                    x,
                    (0, seq_len - cache_size, 0),
                    (batch_size, cache_size, self.width)
                )
            else:
                combined = jnp.concatenate([cache_dict[cache_key], x], axis=1)
                combined_len = combined.shape[1]
                new_cache_state = jax.lax.dynamic_slice(
                    combined,
                    (0, combined_len - cache_size, 0), 
                    (batch_size, cache_size, self.width)
                )
            
            cache_dict[cache_key] = new_cache_state
        
        self.conv_cache.value = cache_dict
        return conv_output

    def reset_conv_cache(self):
        """Reset conv cache for new sequences."""
        if (self.is_mutable_collection('conv_cache') and 
            hasattr(self, 'conv_cache') and 
            isinstance(self.conv_cache.value, dict)):
            cache_dict = self.conv_cache.value
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
            self.reset_all_caches()
        
        B = hidden_states.shape[0]
        self._init_caches_if_needed(B)
        self.config.output_ttt_stats = output_ttt_stats
        
        # Process both fast and slow paths in parallel
        fast_output, fast_stats = self._process_fast_path(
            hidden_states, position_ids, ttt_lr_mult
        )
        
        slow_output, slow_stats = self._process_slow_path(
            hidden_states, position_ids, ttt_lr_mult
        )
        
        # Mix outputs at the final level
        mixed_output = self._mix_outputs(fast_output, slow_output)
        
        # Combine stats (use fast path stats as primary, add slow path info)
        precompute_stats, ttt_loss_mse_init_fast, ttt_loss_mse_step_0_fast, ttt_loss_mse_step_1_fast = fast_stats
        _, ttt_loss_mse_init_slow, ttt_loss_mse_step_0_slow, ttt_loss_mse_step_1_slow = slow_stats
        
        combined_stats = (
            precompute_stats,
            ttt_loss_mse_init_fast,
            ttt_loss_mse_step_0_fast, 
            ttt_loss_mse_step_1_fast,
            # Additional slow path stats for monitoring
            ttt_loss_mse_init_slow,
            ttt_loss_mse_step_0_slow,
            ttt_loss_mse_step_1_slow
        )
        
        return mixed_output, combined_stats

    def _init_caches_if_needed(self, batch_size):
        """Initialize all caches if needed."""
        # Fast path TTT cache
        if (self.is_mutable_collection('ttt_cache_fast') and 
            hasattr(self, 'ttt_cache_fast') and 
            self.ttt_cache_fast.value == ()):
            batched_ttt_params_fast = tree_map(
                lambda p: p[None].repeat(batch_size, axis=0) if isinstance(p, jnp.ndarray) else p, 
                self.ttt_params_fast
            )
            self.ttt_cache_fast.value = batched_ttt_params_fast
        
        # Slow path TTT cache
        if (self.is_mutable_collection('ttt_cache_slow') and 
            hasattr(self, 'ttt_cache_slow') and 
            self.ttt_cache_slow.value == ()):
            batched_ttt_params_slow = tree_map(
                lambda p: p[None].repeat(batch_size, axis=0) if isinstance(p, jnp.ndarray) else p, 
                self.ttt_params_slow
            )
            self.ttt_cache_slow.value = batched_ttt_params_slow
        
        # Conv cache initialization
        self._init_conv_cache_if_needed(batch_size)

    def _process_fast_path(self, hidden_states, position_ids, ttt_lr_mult):
        """Process the fast path with regular mini-batch size."""
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(
            hidden_states, position_ids, use_slow_path=False
        )
        eta *= ttt_lr_mult
        
        Z, ttt_stats = self.ttt_generic(XQ, XK, XV, eta, use_slow_path=False)
        
        Z = self.post_norm(Z)
        Z = self.apply_gate(hidden_states, Z)
        output = self.project_ttt_outputs(Z)
        
        # Update fast cache
        if (self.is_mutable_collection('ttt_cache_fast') and hasattr(self, 'ttt_cache_fast')):
            self.ttt_cache_fast.value = ttt_stats[-1]  # Final params
        
        return output, (precompute_stats,) + ttt_stats[:-1]

    def _process_slow_path(self, hidden_states, position_ids, ttt_lr_mult):
        """Process the slow path with larger mini-batch size."""
        XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(
            hidden_states, position_ids, use_slow_path=True
        )
        eta *= ttt_lr_mult * 0.5  # Slower learning rate for slow path
        
        Z, ttt_stats = self.ttt_generic(XQ, XK, XV, eta, use_slow_path=True)
        
        Z = self.post_norm(Z)
        Z = self.apply_gate(hidden_states, Z)
        output = self.project_ttt_outputs(Z)
        
        # Update slow cache
        if (self.is_mutable_collection('ttt_cache_slow') and hasattr(self, 'ttt_cache_slow')):
            self.ttt_cache_slow.value = ttt_stats[-1]  # Final params
        
        return output, (precompute_stats,) + ttt_stats[:-1]

    def _mix_outputs(self, fast_output, slow_output):
        """Mix fast and slow path outputs using learnable weights."""
        if not self.per_head_mixing:
            # Global mixing
            fast_weight, slow_weight = self.get_mixing_weight()
            mixed_output = fast_weight * fast_output + slow_weight * slow_output
        else:
            # Per-head mixing (reshape for broadcasting)
            B, N, D = fast_output.shape
            fast_output_reshaped = fast_output.reshape(B, N, self.num_heads, self.head_dim)
            slow_output_reshaped = slow_output.reshape(B, N, self.num_heads, self.head_dim)
            
            mixed_heads = []
            for h in range(self.num_heads):
                fast_weight, slow_weight = self.get_mixing_weight(h)
                mixed_head = (fast_weight * fast_output_reshaped[:, :, h, :] + 
                             slow_weight * slow_output_reshaped[:, :, h, :])
                mixed_heads.append(mixed_head)
            
            mixed_output = jnp.stack(mixed_heads, axis=2).reshape(B, N, D)
        
        return mixed_output

    def setup_qkvo(self):
        """Setup Q, K, V, O projections - to be overridden by subclasses."""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )

    def setup_token_idx(self):
        """Setup token indexing for both paths."""
        # Fast path token indices
        self.token_idx_fast = 1.0 / jnp.arange(1, self.mini_batch_size_fast + 1, dtype=jnp.float32)
        self.learnable_token_idx_fast = self.param(
            "learnable_token_idx_fast", nn.initializers.zeros, (self.mini_batch_size_fast,), jnp.float32
        )
        
        # Slow path token indices
        self.token_idx_slow = 1.0 / jnp.arange(1, self.mini_batch_size_slow + 1, dtype=jnp.float32)
        self.learnable_token_idx_slow = self.param(
            "learnable_token_idx_slow", nn.initializers.zeros, (self.mini_batch_size_slow,), jnp.float32
        )

    def setup_ttt_lr_gate(self):
        """Setup TTT learning rate gate."""
        self.learnable_ttt_lr = LinearLayerTemplate(
            width=1, use_bias=True, name="learnable_ttt_lr", dtype=self.dtype, param_dtype=self.param_dtype
        )
        learnable_ttt_lr_params = self.learnable_ttt_lr.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
        self.learnable_ttt_lr_params = get_multi_head_params(
            self,
            learnable_ttt_lr_params,
            param_dtype=self.param_dtype,
            kernel_init="normal",
            std=getattr(self.config, 'initializer_range', 0.02),
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_mini_batches(self, hidden_states, use_slow_path=False):
        """Split into mini-batches according to the specified path."""
        B, N, num_head, head_dim = hidden_states.shape
        
        if use_slow_path:
            seq_shape = self.seq_shape_slow
        else:
            seq_shape = self.seq_shape_fast
        
        hidden_states = hidden_states.reshape(B, *seq_shape, self.num_heads, self.head_dim).transpose(
            0, 3, 1, 2, 4
        )
        return hidden_states

    def get_qkv_projections(self, batch):
        """Default implementation - can be overridden in subclasses for conv caching."""
        XQ, XK, XV = self.wq(batch), self.wk(batch), self.wv(batch)
        return XQ, XK, XV

    def get_eta(self, X, use_slow_path=False):
        """Get learning rate for the specified path."""
        if use_slow_path:
            mini_batch_size = self.mini_batch_size_slow
            token_idx = self.token_idx_slow
            learnable_token_idx = self.learnable_token_idx_slow
        else:
            mini_batch_size = self.mini_batch_size_fast
            token_idx = self.token_idx_fast
            learnable_token_idx = self.learnable_token_idx_fast
        
        # Use existing logic but with appropriate mini-batch size
        learnable_ttt_lr = vmap(
            lambda x, p: self.learnable_ttt_lr.apply({"params": p}, x), 
            axis_name="head", in_axes=[None, 0], out_axes=1
        )(X, self.learnable_ttt_lr_params)
        learnable_ttt_lr = nn.sigmoid(learnable_ttt_lr)
        learnable_ttt_lr = learnable_ttt_lr.transpose(0, 1, 2, 4, 3)

        token_idx = learnable_token_idx + token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        eta = (
            (getattr(self.config, 'ttt_base_lr', 1.0) * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_ttt_lr
            / self.head_dim
        )
        return eta

    def get_ttt_inputs(self, batch, position_ids, use_slow_path=False):
        """Get TTT inputs for either fast or slow path."""
        B, N, F = batch.shape
        
        if use_slow_path:
            mini_batch_size = self.mini_batch_size_slow
            seq_shape = self.seq_shape_slow
            freqs_cis = self.freqs_cis_slow
        else:
            mini_batch_size = self.mini_batch_size_fast
            seq_shape = self.seq_shape_fast
            freqs_cis = self.freqs_cis_fast
        
        n_mini_batch = N // mini_batch_size
        X = batch.reshape(B, *seq_shape, self.width)

        XQ, XK, XV = self.get_qkv_projections(batch)

        # Stats computation
        if self.config.output_ttt_stats:
            XV_last_in_mini_batch = XV[:, ::mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            XK_last_in_mini_batch = XK[:, ::mini_batch_size, ...].reshape(
                B, n_mini_batch, self.num_heads, self.head_dim
            )
            ssl_tgt_last_in_mini_batch = XV_last_in_mini_batch - XK_last_in_mini_batch
            ssl_tgt_mean = (XV - XK).mean(axis=1, keepdims=True).reshape(B, 1, self.num_heads, self.head_dim)
            ssl_tgt_last_in_mini_batch_from_mean_mse = ((ssl_tgt_last_in_mini_batch - ssl_tgt_mean) ** 2).mean(
                axis=(0, 2, 3)
            )
        else:
            ssl_tgt_last_in_mini_batch_from_mean_mse = None

        # Apply sharding constraints
        XQ = with_sharding_constraint(XQ, PS(("dp", "fsdp"), None, "mp"))
        XK = with_sharding_constraint(XK, PS(("dp", "fsdp"), None, "mp"))
        XV = with_sharding_constraint(XV, PS(("dp", "fsdp"), None, "mp"))

        # Split heads
        XQ = self._split_heads(XQ)
        XK = self._split_heads(XK)
        XV = self._split_heads(XV)

        # Apply RoPE
        freqs_cis_selected = jnp.take(freqs_cis, position_ids % mini_batch_size, axis=0)
        XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis_selected, dtype=self.dtype)

        # Split mini-batches according to path
        XQ = self._split_mini_batches(XQ, use_slow_path)
        XK = self._split_mini_batches(XK, use_slow_path)
        XV = self._split_mini_batches(XV, use_slow_path)

        eta = self.get_eta(X, use_slow_path)

        return (XQ, XK, XV, eta, ssl_tgt_last_in_mini_batch_from_mean_mse)

    def apply_gate(self, hidden_states, ttt_output):
        """Apply gating - to be overridden by subclasses."""
        return ttt_output

    def project_ttt_outputs(self, XQW_batch):
        """Project TTT outputs."""
        z_batch = self.wo(XQW_batch)
        return z_batch

    def ttt_generic(self, XQ, XK, XV, eta, use_slow_path=False):
        """Generic TTT processing that works for both paths."""
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta, ttt_params=None):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, ttt_params_init, ttt_norm_params):
                def compute_mini_batch(carry, inputs):
                    ttt_params_mini_batch_init = carry
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
                
                final_carry, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch, getattr(self.config, 'remat_mini_batch_group_size', 1), ttt_params_init, inputs
                )
                
                Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1 = outputs

                return (Z.reshape(-1, self.head_dim), ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, final_carry)

            # Get appropriate cached parameters
            if use_slow_path:
                params_to_use = self.ttt_params_slow if ttt_params is None else ttt_params
            else:
                params_to_use = self.ttt_params_fast if ttt_params is None else ttt_params

            outputs = parallelize_over_heads(
                XQ, XK, XV, eta, params_to_use, self.ttt_norm_params
            )
            return outputs

        # Get cached parameters
        if use_slow_path:
            ttt_params_cache = (self.ttt_cache_slow.value 
                              if hasattr(self, 'ttt_cache_slow') and self.ttt_cache_slow.value != () 
                              else None)
        else:
            ttt_params_cache = (self.ttt_cache_fast.value 
                              if hasattr(self, 'ttt_cache_fast') and self.ttt_cache_fast.value != () 
                              else None)
        
        outputs = update_embed(XQ, XK, XV, eta, ttt_params=ttt_params_cache)
        Z, ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, ttt_params_final = outputs

        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, -1)

        if self.config.output_ttt_stats:
            ttt_loss_mse_init = ttt_loss_mse_init.mean(axis=(0, 1))
            ttt_loss_mse_step_0 = ttt_loss_mse_step_0.mean(axis=(0, 1))
            ttt_loss_mse_step_1 = ttt_loss_mse_step_1.mean(axis=(0, 1))

        return Z, (ttt_loss_mse_init, ttt_loss_mse_step_0, ttt_loss_mse_step_1, ttt_params_final)

    def reset_all_caches(self):
        """Reset all caches for new sequences."""
        # Reset conv cache
        self.reset_conv_cache()
        
        # Reset TTT caches
        if self.is_mutable_collection('ttt_cache_fast') and hasattr(self, 'ttt_cache_fast'):
            self.ttt_cache_fast.value = ()
        if self.is_mutable_collection('ttt_cache_slow') and hasattr(self, 'ttt_cache_slow'):
            self.ttt_cache_slow.value = ()

    def process_mini_batch(self, *args, **kwargs):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_mini_batch")


class TTTLinearBase(TTTBase):
    """Base class for TTT Linear implementations."""
    
    def setup(self):
        super().setup()

        # Separate parameters for fast and slow paths
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        self.W1_fast = self.param(
            "ttt_dense_0_fast",
            nn.initializers.normal(initializer_range),
            (self.num_heads, self.head_dim, self.head_dim),
            self.param_dtype,
        )
        
        self.W1_slow = self.param(
            "ttt_dense_0_slow", 
            nn.initializers.normal(initializer_range),
            (self.num_heads, self.head_dim, self.head_dim),
            self.param_dtype,
        )
        
        self.ttt_params_fast = (self.W1_fast,)
        self.ttt_params_slow = (self.W1_slow,)

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
        mini_batch_size = XQ_mini_batch.shape[0]  # Dynamically get mini-batch size
        square_eta_mini_batch = eta_mini_batch[:mini_batch_size]
        last_eta_in_mini_batch = eta_mini_batch[-1][:, None]

        X1 = XK_mini_batch
        Z1 = X1 @ W1_init
        ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: self.ttt_norm.apply({"params": ttt_norm_params}, z), Z1)
        ssl_target = XV_mini_batch - XK_mini_batch
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0]

        # Calculate TTT loss using W_init of the current mini-batch
        if getattr(self.config, 'output_ttt_stats', False):
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate TTT loss using W_init of the entire sequence
        if getattr(self.config, 'output_ttt_stats', False):
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
        if getattr(self.config, 'output_ttt_stats', False):
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
    """TTT Linear with dual-path processing and gating."""
    
    def setup(self):
        super().setup()
        
        # Gate for mixing
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        """Setup shared Q/K projection with separate convolutions."""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        
        # Shared Q/K projection
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        
        # Separate convolutions for Q and K
        remat_conv = getattr(self.config, 'remat_conv', "")
        if remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
            
        conv_width = getattr(self.config, 'conv_width', 4)
        self.conv_q = conv_module(
            self.config.hidden_size,
            (conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (conv_width,),
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
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        
        # Output projection
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
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
        """Apply gating mechanism."""
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output


class TTTMLPBase(TTTBase):
    """Base class for TTT MLP implementations."""
    
    def setup(self):
        super().setup()
        
        # Separate parameters for fast and slow paths
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        self.W1_fast = self.param(
            "ttt_dense_0_fast",
            nn.initializers.normal(initializer_range),
            (self.num_heads, self.head_dim, 4 * self.head_dim),
            self.param_dtype,
        )
        self.W2_fast = self.param(
            "ttt_dense_1_fast",
            nn.initializers.normal(initializer_range),
            (self.num_heads, 4 * self.head_dim, self.head_dim),
            self.param_dtype,
        )
        
        self.W1_slow = self.param(
            "ttt_dense_0_slow",
            nn.initializers.normal(initializer_range),
            (self.num_heads, self.head_dim, 4 * self.head_dim),
            self.param_dtype,
        )
        self.W2_slow = self.param(
            "ttt_dense_1_slow",
            nn.initializers.normal(initializer_range),
            (self.num_heads, 4 * self.head_dim, self.head_dim),
            self.param_dtype,
        )
        
        self.ttt_params_fast = (self.W1_fast, self.W2_fast)
        self.ttt_params_slow = (self.W1_slow, self.W2_slow)

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
        mini_batch_size = XQ_mini_batch.shape[0]  # Dynamically get mini-batch size
        square_eta_mini_batch = eta_mini_batch[:mini_batch_size]
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

        if getattr(self.config, 'output_ttt_stats', False):
            ttt_loss_mse_step_0 = (grad_l_wrt_ttt_norm_out[-1] ** 2).mean()
        else:
            ttt_loss_mse_step_0 = None

        # Calculate ttt loss using W_init of the entire sequence
        if getattr(self.config, 'output_ttt_stats', False):
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

        if getattr(self.config, 'output_ttt_stats', False):
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
    """TTT MLP with dual-path processing and gating."""
    
    def setup(self):
        super().setup()
        
        # Gate for mixing
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        self.wg = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )

    def setup_qkvo(self):
        """Setup shared Q/K projection with separate convolutions."""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        
        # Shared Q/K projection
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        
        # Separate convolutions for Q and K with caching support
        remat_conv = getattr(self.config, 'remat_conv', "")
        if remat_conv != "":
            conv_module = nn_partitioning.remat(
                nn.Conv, policy=get_gradient_checkpoint_policy(remat_conv), prevent_cse=True
            )
        else:
            conv_module = nn.Conv
            
        conv_width = getattr(self.config, 'conv_width', 4)
        self.conv_q = conv_module(
            self.config.hidden_size,
            (conv_width,),
            padding="CAUSAL",
            feature_group_count=self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.conv_k = conv_module(
            self.config.hidden_size,
            (conv_width,),
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
            kernel_init=jax.nn.initializers.normal(initializer_range),
            precision=self.precision,
        )
        
        # Output projection
        self.wo = nn.Dense(
            self.width,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(initializer_range),
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
        """Apply gating mechanism."""
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * ttt_output
        return output