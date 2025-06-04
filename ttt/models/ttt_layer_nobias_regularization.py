"""
JAX Lattice-TTT Implementation with Dual Form

Key Innovation: First-order approximation of normalization constraints to enable dual form.

Mathematical approach:
1. Linearize normalize_columns(S + ΔS) ≈ normalize_columns(S) + Jacobian_norm(S) @ ΔS
2. This preserves linear structure: Φ_t ≈ Φ_0 + Jacobian @ (-η * Σ G_i)
3. Enables parallel computation via triangular masking
4. Maintains orthogonal update properties approximately

Trade-off: Exact orthogonality → Approximate orthogonality for computational efficiency
"""

import math
import numpy as np
from functools import partial
from typing import Any, Union, Sequence, Optional, Tuple

import jax
import jax.numpy as jnp
import flax
from jax import vmap
from jax.tree_util import tree_map
from jax.sharding import PartitionSpec as PS
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from ttt.infra.jax_utils import with_sharding_constraint, get_gradient_checkpoint_policy

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


def normalize_columns(S):
    """Normalize each column of matrix S to unit norm."""
    norms = jnp.linalg.norm(S, axis=0, keepdims=True)
    norms = jnp.where(norms < 1e-8, 1.0, norms)
    return S / norms


def compute_normalization_jacobian(S):
    """
    Compute Jacobian of normalize_columns operation.
    
    For each column s_i, ∂φ_i/∂s_i = (1/||s_i||) * [I - φ_i φ_i^T]
    where φ_i = s_i / ||s_i|| is the normalized column.
    
    Returns: Function that applies Jacobian to a perturbation ΔS
    """
    d, m = S.shape
    norms = jnp.linalg.norm(S, axis=0, keepdims=True)
    norms = jnp.where(norms < 1e-8, 1.0, norms)
    phi = S / norms  # Normalized columns
    
    def apply_jacobian(delta_S):
        """Apply normalization Jacobian to perturbation delta_S."""
        # For each column i: (1/||s_i||) * [I - φ_i φ_i^T] @ Δs_i
        result = jnp.zeros_like(delta_S)
        for i in range(m):
            phi_i = phi[:, i:i+1]  # Keep as column vector
            norm_i = norms[0, i]
            delta_s_i = delta_S[:, i:i+1]
            
            # Orthogonal projection: [I - φ_i φ_i^T] @ Δs_i
            proj_delta = delta_s_i - phi_i * (phi_i.T @ delta_s_i)
            result = result.at[:, i:i+1].set(proj_delta / norm_i)
        
        return result
    
    return apply_jacobian


def dual_form_lattice_linear(W0, X, Y, eta_sequence, lattice_norm_apply, mask):
    """
    Dual form implementation for Lattice Linear layer.
    
    Key insight: Linearize normalization around W0 to maintain linear structure.
    
    Approximation: normalize_columns(W0 - η*Σ G_i) ≈ Φ_0 - η*Σ (J_norm @ G_i)
    where Φ_0 = normalize_columns(W0) and J_norm is normalization Jacobian.
    """
    b, d = X.shape
    
    # Step 1: Compute initial normalized weights and Jacobian
    Phi_0 = normalize_columns(W0)
    jacobian_apply = compute_normalization_jacobian(W0)
    
    # Step 2: Define loss function for gradient computation
    def single_token_loss(phi_normalized, x, y):
        """Loss for single token with already normalized weights."""
        z = x @ phi_normalized
        z_norm = lattice_norm_apply(z)
        error = z_norm - y
        return 0.5 * jnp.sum(error ** 2)
    
    # Step 3: Compute gradients w.r.t. normalized weights at Φ_0
    grad_phi_fn = jax.grad(single_token_loss, argnums=0)
    
    # Compute all gradients w.r.t. normalized weights in parallel
    grad_phi_batch = jax.vmap(grad_phi_fn, in_axes=(None, 0, 0))(Phi_0, X, Y)
    
    # Step 4: Map gradients back to unnormalized space using Jacobian^T
    # If φ = f(W) and ∂L/∂φ is known, then ∂L/∂W = J^T @ (∂L/∂φ)
    # For orthogonal projection Jacobian, J^T = J, so we can reuse jacobian_apply
    grad_W_batch = jax.vmap(jacobian_apply)(grad_phi_batch)
    
    # Step 5: Compute final weights using cumulative sum
    eta_expanded = eta_sequence[:, None, None]  # Shape: (b, 1, 1)
    weighted_grads = eta_expanded * grad_W_batch  # Shape: (b, d, m)
    cumulative_grads = jnp.cumsum(weighted_grads, axis=0)  # Shape: (b, d, m)
    
    # Final weights (still unnormalized for now)
    W_final = W0 - cumulative_grads[-1]
    
    # Step 6: Compute all outputs using dual form with first-order approximation
    # Φ_t ≈ Φ_0 - cumulative_grads_normalized[t-1]
    cumulative_grads_normalized = jax.vmap(jacobian_apply)(cumulative_grads)
    
    # Approximate normalized weights at each step
    Phi_approx = Phi_0[None, :, :] - cumulative_grads_normalized  # Shape: (b, d, m)
    
    # Compute outputs in parallel
    Z_raw = jnp.einsum('bdi,bi->bd', Phi_approx, X)  # Batch matrix multiply
    Z_normalized = jax.vmap(lattice_norm_apply)(Z_raw)
    
    return Z_normalized, W_final


def dual_form_lattice_mlp(W1_0, W2_0, X, Y, eta_sequence, lattice_norm_apply, gelu_fn, mask):
    """
    Dual form implementation for Lattice MLP layer.
    
    More complex due to two weight matrices and nonlinearity.
    Uses first-order approximations for both normalizations.
    """
    b, d = X.shape
    hidden_dim = W1_0.shape[1]
    
    # Step 1: Compute initial normalized weights and Jacobians
    Phi1_0 = normalize_columns(W1_0)
    Phi2_0 = normalize_columns(W2_0)
    jacobian1_apply = compute_normalization_jacobian(W1_0)
    jacobian2_apply = compute_normalization_jacobian(W2_0)
    
    # Step 2: Define loss function for single token
    def single_token_mlp_loss(phi1_norm, phi2_norm, x, y):
        """Loss for single token with normalized weights."""
        z1 = x @ phi1_norm
        a1 = gelu_fn(z1)
        z2 = a1 @ phi2_norm
        z_norm = lattice_norm_apply(z2)
        error = z_norm - y
        return 0.5 * jnp.sum(error ** 2)
    
    # Step 3: Compute gradients w.r.t. both normalized weight matrices
    grad_fn = jax.grad(single_token_mlp_loss, argnums=(0, 1))
    
    # Compute gradients for all tokens in parallel
    grad_phi1_batch, grad_phi2_batch = jax.vmap(
        grad_fn, in_axes=(None, None, 0, 0)
    )(Phi1_0, Phi2_0, X, Y)
    
    # Step 4: Map gradients back to unnormalized space
    grad_W1_batch = jax.vmap(jacobian1_apply)(grad_phi1_batch)
    grad_W2_batch = jax.vmap(jacobian2_apply)(grad_phi2_batch)
    
    # Step 5: Compute cumulative updates
    eta_expanded = eta_sequence[:, None, None]
    
    weighted_grads1 = eta_expanded * grad_W1_batch
    weighted_grads2 = eta_expanded * grad_W2_batch
    
    cumulative_grads1 = jnp.cumsum(weighted_grads1, axis=0)
    cumulative_grads2 = jnp.cumsum(weighted_grads2, axis=0)
    
    # Final weights
    W1_final = W1_0 - cumulative_grads1[-1]
    W2_final = W2_0 - cumulative_grads2[-1]
    
    # Step 6: Compute outputs using first-order approximations
    cumulative_grads1_normalized = jax.vmap(jacobian1_apply)(cumulative_grads1)
    cumulative_grads2_normalized = jax.vmap(jacobian2_apply)(cumulative_grads2)
    
    # Approximate normalized weights at each step
    Phi1_approx = Phi1_0[None, :, :] - cumulative_grads1_normalized
    Phi2_approx = Phi2_0[None, :, :] - cumulative_grads2_normalized
    
    # Forward pass in parallel
    Z1_raw = jnp.einsum('bdi,bi->bd', Phi1_approx, X)
    A1 = jax.vmap(gelu_fn)(Z1_raw)
    Z2_raw = jnp.einsum('bdi,bi->bd', Phi2_approx, A1)
    Z_normalized = jax.vmap(lattice_norm_apply)(Z2_raw)
    
    return Z_normalized, (W1_final, W2_final)


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


class LatticeBase(nn.Module):
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
        self.setup_lattice_lr_gate()

        # Lattice normalization layer
        self.lattice_norm = LayerNormTemplate(dtype=self.dtype, param_dtype=self.param_dtype)
        lattice_norm_params = self.lattice_norm.init(jax.random.PRNGKey(0), jnp.ones([1, self.head_dim]))["params"]
        self.lattice_norm_params = get_multi_head_params(
            self, lattice_norm_params, param_dtype=self.param_dtype, kernel_init="layer_norm"
        )
        self.post_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)

        self.lattice_params = ()

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

    def setup_lattice_lr_gate(self):
        self.learnable_lattice_lr = LinearLayerTemplate(
            width=1, use_bias=True, name="learnable_lattice_lr", dtype=self.dtype, param_dtype=self.param_dtype
        )
        learnable_lattice_lr_params = self.learnable_lattice_lr.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
        self.learnable_lattice_lr_params = get_multi_head_params(
            self,
            learnable_lattice_lr_params,
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
        XQ, XK, XV = self.wq(batch), self.wk(batch), self.wv(batch)
        return XQ, XK, XV

    def get_eta(self, X):
        learnable_lattice_lr = vmap(
            lambda x, p: self.learnable_lattice_lr.apply({"params": p}, x), axis_name="head", in_axes=[None, 0], out_axes=1
        )(X, self.learnable_lattice_lr_params)
        learnable_lattice_lr = nn.sigmoid(learnable_lattice_lr)
        learnable_lattice_lr = learnable_lattice_lr.transpose(0, 1, 2, 4, 3)

        token_idx = self.learnable_token_idx + self.token_idx
        token_idx = jnp.clip(token_idx, a_min=0.0)

        eta = (
            (self.config.lattice_base_lr * token_idx).reshape(1, 1, 1, token_idx.shape[0], -1)
            * learnable_lattice_lr
            / self.head_dim
        )
        return eta

    def get_lattice_inputs(self, batch, position_ids):
        B, N, F = batch.shape
        n_mini_batch = N // self.mini_batch_size
        X = batch.reshape(B, *self.seq_shape, self.width)

        XQ, XK, XV = self.get_qkv_projections(batch)

        if self.config.output_lattice_stats:
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

        return (XQ, XK, XV, eta, (ssl_tgt_last_in_mini_batch_from_mean_mse,))

    def apply_gate(self, hidden_states, lattice_output):
        return lattice_output

    def project_lattice_outputs(self, XQW_batch):
        z_batch = self.wo(XQW_batch)
        return z_batch

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        lattice_params_init,
        lattice_params_mini_batch_init,
        lattice_norm_params,
    ):
        raise NotImplementedError

    def lattice_update(self, XQ, XK, XV, eta, input_ids):
        B, N = XV.shape[0], XV.shape[2] * XV.shape[3]

        @partial(vmap, axis_name="batch")
        def update_embed(XQ, XK, XV, eta):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(XQ, XK, XV, eta, lattice_params_init, lattice_norm_params):
                def compute_mini_batch(lattice_params_mini_batch_init, inputs):
                    XQ_mini_batch = inputs["XQ"]
                    XK_mini_batch = inputs["XK"]
                    XV_mini_batch = inputs["XV"]
                    eta_mini_batch = inputs["eta"]

                    lattice_params_last_in_mini_batch, outputs = self.process_mini_batch(
                        XQ_mini_batch,
                        XK_mini_batch,
                        XV_mini_batch,
                        eta_mini_batch,
                        lattice_params_init,
                        lattice_params_mini_batch_init,
                        lattice_norm_params,
                    )
                    return lattice_params_last_in_mini_batch, outputs

                inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}

                _, outputs = scan_remat_every_n_iterations_scan(
                    compute_mini_batch, self.config.remat_mini_batch_group_size, lattice_params_init, inputs
                )
                Z, lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1 = outputs
                return (Z.reshape(-1, self.head_dim), lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1)

            outputs = parallelize_over_heads(XQ, XK, XV, eta, self.lattice_params, self.lattice_norm_params)
            return outputs

        outputs = update_embed(XQ, XK, XV, eta)
        Z, lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1 = outputs
        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, -1)

        if self.config.output_lattice_stats:
            lattice_loss_mse_init = lattice_loss_mse_init.mean(axis=(0, 1))
            lattice_loss_mse_step_0 = lattice_loss_mse_step_0.mean(axis=(0, 1))
            lattice_loss_mse_step_1 = lattice_loss_mse_step_1.mean(axis=(0, 1))

        return Z, (lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1)

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        position_ids=None,
        deterministic: bool = True,
        output_lattice_stats: bool = False,
        lattice_lr_mult=1.0,
    ):
        self.config.output_lattice_stats = output_lattice_stats
        del deterministic
        XQ, XK, XV, eta, precompute_stats = self.get_lattice_inputs(hidden_states, position_ids=position_ids)
        eta *= lattice_lr_mult
        Z, lattice_stats = self.lattice_update(XQ, XK, XV, eta, input_ids)
        Z = self.post_norm(Z)
        Z = self.apply_gate(hidden_states, Z)
        lattice_output = self.project_lattice_outputs(Z)
        return lattice_output, (*precompute_stats, *lattice_stats)


class LatticeLinearBase(LatticeBase):
    def setup(self):
        super().setup()
        # Initialize with orthonormal columns
        self.W1 = self.param(
            "lattice_dense_0",
            self.orthonormal_init,
            (self.num_heads, self.head_dim, self.head_dim),
            self.param_dtype,
        )
        self.lattice_params = (self.W1,)

    def orthonormal_init(self, key, shape, dtype=jnp.float32):
        """Initialize with orthonormal columns."""
        num_heads, d, m = shape
        matrices = []
        keys = jax.random.split(key, num_heads)
        
        for i in range(num_heads):
            A = jax.random.normal(keys[i], (d, m), dtype=dtype)
            Q, R = jnp.linalg.qr(A)
            Q = Q * jnp.sign(jnp.diag(R))[None, :]
            matrices.append(Q)
        
        return jnp.stack(matrices, axis=0)

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        lattice_params_init,
        lattice_params_mini_batch_init,
        lattice_norm_params,
    ):
        (W1_init,) = lattice_params_mini_batch_init
        eta_sequence = eta_mini_batch[: self.mini_batch_size]
        
        X1 = XK_mini_batch
        ssl_target = XV_mini_batch - XK_mini_batch
        
        # Create lattice norm apply function
        def lattice_norm_apply(z):
            return self.lattice_norm.apply({"params": lattice_norm_params}, z)
        
        # Create triangular mask for dual form
        mask = jnp.tril(jnp.ones((self.mini_batch_size, self.mini_batch_size)))
        
        # Use dual form with first-order approximation
        Z_normalized, W1_final = dual_form_lattice_linear(
            W1_init, X1, ssl_target, eta_sequence, lattice_norm_apply, mask
        )
        
        # Compute output for mini-batch
        X1_bar = XQ_mini_batch
        W1_final_normalized = normalize_columns(W1_final)
        Z1_bar = X1_bar @ W1_final_normalized
        lattice_norm_out_bar = lattice_norm_apply(Z1_bar)
        output_mini_batch = X1_bar + lattice_norm_out_bar
        
        # Statistics computation (simplified for dual form)
        if self.config.output_lattice_stats:
            # Initial loss
            (W1_0,) = lattice_params_init
            W1_0_normalized = normalize_columns(W1_0)
            Z1_0 = X1 @ W1_0_normalized
            lattice_norm_out_0 = lattice_norm_apply(Z1_0)
            lattice_loss_mse_init = ((lattice_norm_out_0 - ssl_target)[-1] ** 2).mean()
            
            # Step 0 loss (current mini-batch start)
            W1_init_normalized = normalize_columns(W1_init)
            Z1_init = X1 @ W1_init_normalized
            lattice_norm_out_init = lattice_norm_apply(Z1_init)
            lattice_loss_mse_step_0 = ((lattice_norm_out_init - ssl_target)[-1] ** 2).mean()
            
            # Step 1 loss (after update)
            X1_last_fwd_new = X1[-1:] @ W1_final_normalized
            X1_last_fwd_new = lattice_norm_apply(X1_last_fwd_new)
            lattice_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            lattice_loss_mse_init = None
            lattice_loss_mse_step_0 = None
            lattice_loss_mse_step_1 = None

        lattice_params_mini_batch_new = (W1_final,)

        return (
            lattice_params_mini_batch_new,
            (output_mini_batch, lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1),
        )


class LatticeLinear(LatticeLinearBase):
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
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
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

    def get_qkv_projections(self, batch):
        xqk, XV = self.wq(batch), self.wv(batch)
        XQ = self.conv_q(xqk)
        XK = self.conv_k(xqk)
        return XQ, XK, XV

    def apply_gate(self, hidden_states, lattice_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * lattice_output
        return output


class LatticeMLPBase(LatticeBase):
    def setup(self):
        super().setup()
        # Initialize with orthonormal matrices
        self.W1 = self.param(
            "lattice_dense_0",
            self.orthonormal_init,
            (self.num_heads, self.head_dim, 4 * self.head_dim),
            self.param_dtype,
        )
        self.W2 = self.param(
            "lattice_dense_1",
            self.orthonormal_init,
            (self.num_heads, 4 * self.head_dim, self.head_dim),
            self.param_dtype,
        )
        self.lattice_params = (self.W1, self.W2)

    def orthonormal_init(self, key, shape, dtype=jnp.float32):
        """Initialize with orthonormal columns."""
        num_heads, d_in, d_out = shape
        matrices = []
        keys = jax.random.split(key, num_heads)
        
        for i in range(num_heads):
            A = jax.random.normal(keys[i], (d_in, d_out), dtype=dtype)
            Q, R = jnp.linalg.qr(A)
            Q = Q * jnp.sign(jnp.diag(R))[None, :]
            matrices.append(Q)
        
        return jnp.stack(matrices, axis=0)

    def process_mini_batch(
        self,
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        lattice_params_init,
        lattice_params_mini_batch_init,
        lattice_norm_params,
    ):
        W1_init, W2_init = lattice_params_mini_batch_init
        eta_sequence = eta_mini_batch[: self.mini_batch_size]
        
        X1 = XK_mini_batch
        ssl_target = XV_mini_batch - X1
        
        # Create function closures
        def lattice_norm_apply(z):
            return self.lattice_norm.apply({"params": lattice_norm_params}, z)
        
        def gelu_fn(x):
            return nn.gelu(x)
        
        # Create triangular mask
        mask = jnp.tril(jnp.ones((self.mini_batch_size, self.mini_batch_size)))
        
        # Use dual form with first-order approximation
        Z_normalized, (W1_final, W2_final) = dual_form_lattice_mlp(
            W1_init, W2_init, X1, ssl_target, eta_sequence, lattice_norm_apply, gelu_fn, mask
        )
        
        # Compute output for mini-batch
        X1_bar = XQ_mini_batch
        W1_final_normalized = normalize_columns(W1_final)
        W2_final_normalized = normalize_columns(W2_final)
        
        Z1_bar = X1_bar @ W1_final_normalized
        X2_bar = nn.gelu(Z1_bar)
        Z2_bar = X2_bar @ W2_final_normalized
        lattice_norm_out_bar = lattice_norm_apply(Z2_bar)
        output_mini_batch = X1_bar + lattice_norm_out_bar
        
        # Statistics computation
        if self.config.output_lattice_stats:
            # Initial loss
            W1_0, W2_0 = lattice_params_init
            W1_0_normalized = normalize_columns(W1_0)
            W2_0_normalized = normalize_columns(W2_0)
            Z1_0 = X1 @ W1_0_normalized
            X2_0 = nn.gelu(Z1_0)
            Z2_0 = X2_0 @ W2_0_normalized
            lattice_norm_out_0 = lattice_norm_apply(Z2_0)
            lattice_loss_mse_init = ((lattice_norm_out_0 - ssl_target)[-1] ** 2).mean()
            
            # Step 0 loss
            W1_init_normalized = normalize_columns(W1_init)
            W2_init_normalized = normalize_columns(W2_init)
            Z1_init = X1 @ W1_init_normalized
            X2_init = nn.gelu(Z1_init)
            Z2_init = X2_init @ W2_init_normalized
            lattice_norm_out_init = lattice_norm_apply(Z2_init)
            lattice_loss_mse_step_0 = ((lattice_norm_out_init - ssl_target)[-1] ** 2).mean()
            
            # Step 1 loss
            X1_last_fwd_new = nn.gelu(X1[-1:] @ W1_final_normalized) @ W2_final_normalized
            X1_last_fwd_new = lattice_norm_apply(X1_last_fwd_new)
            lattice_loss_mse_step_1 = ((X1_last_fwd_new - ssl_target[-1:]) ** 2).mean()
        else:
            lattice_loss_mse_init = None
            lattice_loss_mse_step_0 = None
            lattice_loss_mse_step_1 = None

        lattice_params_mini_batch_new = (W1_final, W2_final)

        return (
            lattice_params_mini_batch_new,
            (output_mini_batch, lattice_loss_mse_init, lattice_loss_mse_step_0, lattice_loss_mse_step_1),
        )


class LatticeMLP(LatticeMLPBase):
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
        self.wq = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
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

    def get_qkv_projections(self, batch):
        xqk, XV = self.wq(batch), self.wv(batch)
        XQ = self.conv_q(xqk)
        XK = self.conv_k(xqk)
        return XQ, XK, XV

    def apply_gate(self, hidden_states, lattice_output):
        y = self.wg(hidden_states)
        y = nn.gelu(y)
        output = y * lattice_output
        return output