import mlxu
import wandb
import os
import os.path as osp
import json


from tqdm import tqdm
from copy import deepcopy


import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.experimental.multihost_utils import process_allgather
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict


from ttt.infra.optimizers import OptimizerFactory
from ttt.dataloader.language_modeling_hf import LMDataModule
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.models.model import ModelConfig, CausalLM
from ttt.infra.jax_utils import (
JaxRNG,
JaxDistributedConfig,
next_rng,
match_partition_rules,
cross_entropy_loss_and_accuracy,
global_norm,
get_float_dtype_by_name,
set_random_seed,
average_metrics,
get_weight_decay_mask,
make_shard_and_gather_fns,
with_sharding_constraint,
master_print,
log_ttt_stats,
)



FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
seed=0,
mesh_dim="-1,64,1",
dtype="fp32",
eval_mode=False,
load_part="trainstate",
total_steps=100,
load_model_config="",
update_model_config="",
save_checkpoint_freq=100,
save_milestone_freq=0,
dataset_path="",
dataset_name="SaylorTwift/the_pile_books3_minus_gutenberg",
tokenizer_name="meta-llama/Llama-2-7b-hf",
seq_length=2048,
global_batch_size=1,
accum_steps=1,
loader_workers=48,
optimizer=OptimizerFactory.get_default_config(),
checkpointer=StreamingCheckpointer.get_default_config(),
exp_dir="",
exp_name="",
resume_exp_name="",
resume_step="",
jax_distributed=JaxDistributedConfig.get_default_config(),
is_rollback_reshuffle=False,
# Zero-order training options
use_zero_order_training=False,
zero_order_num_chunks=2, # number of chunks to simulate a longer sequence
zero_order_num_perturbations=64,
zero_order_perturbation_scale=1e-3,
zero_order_frequency=0, # 0=never, 1=always, N=every N steps
zero_order_verbose=True,
zero_order_max_grad_norm=1.0,
# Debug options
zero_order_debug_cosine=False,  # Enable cosine similarity debugging
)


def compute_cosine_similarity(grad1, grad2):
    """Compute cosine similarity between two gradient pytrees."""
    # Flatten both gradients
    flat_grad1, _ = jax.flatten_util.ravel_pytree(grad1)
    flat_grad2, _ = jax.flatten_util.ravel_pytree(grad2)
    
    # Compute norms
    norm1 = jnp.linalg.norm(flat_grad1)
    norm2 = jnp.linalg.norm(flat_grad2)
    
    # Compute cosine similarity
    dot_product = jnp.dot(flat_grad1, flat_grad2)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)  # Add epsilon for numerical stability
    
    return cosine_sim


def make_train_step_fn(model, optimizer_info, model_config, accum_steps=1):
    """Original gradient-based training step function."""
    if accum_steps == 1:
        def train_step(train_state, rng, batch, ttt_lr_mult, output_ttt_stats=False):
            rng_generator = JaxRNG(rng)
            batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))


            def loss_and_accuracy(params):
                outputs = model.apply(
                    {'params': params},
                    batch["input_tokens"],
                    ttt_lr_mult=ttt_lr_mult,
                    deterministic=False,
                    output_ttt_stats=output_ttt_stats,
                    rngs=rng_generator(model_config.rng_keys()),
                    )
                logits = outputs.logits
                ttt_stats = outputs.ttt_stats
                loss, _ = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
                return loss, ttt_stats


            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, ttt_stats), grads = grad_fn(train_state.params)


            train_state = train_state.apply_gradients(grads=grads)
            learning_rate = optimizer_info["learning_rate_schedule"](train_state.step)
            grads_norm = global_norm(grads)


            return (train_state, loss, ttt_stats, grads_norm, learning_rate, rng_generator())


    elif accum_steps > 1:
        def train_step(train_state, rng, batch, ttt_lr_mult, output_ttt_stats=False):
            rng_generator = JaxRNG(rng)
            rngs = rng_generator(model_config.rng_keys())


            def computation(carry, micro_batch):
                sum_grads = carry["sum_grads"]
                micro_batch = with_sharding_constraint(micro_batch, PS(("dp", "fsdp")))


                def loss_and_accuracy(params):
                    outputs = model.apply(
                        {'params': params},
                        micro_batch["input_tokens"],
                        ttt_lr_mult=ttt_lr_mult,
                        deterministic=False,
                        output_ttt_stats=output_ttt_stats,
                        rngs=rngs,
                        )
                    logits = outputs.logits
                    ttt_stats = outputs.ttt_stats
                    loss, _ = cross_entropy_loss_and_accuracy(
                        logits, micro_batch["target_tokens"], micro_batch["loss_masks"]
                    )
                    return loss, ttt_stats


                grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
                (loss, ttt_stats), grads = grad_fn(train_state.params)
                sum_grads = tree_map(lambda x, y: x + y, sum_grads, grads)
                carry_new = {"sum_grads": sum_grads}
                return carry_new, (loss, ttt_stats)


            sum_grads = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), train_state.params)
            carry_init = {"sum_grads": sum_grads}
            batch = tree_map(lambda x: x.reshape(FLAGS.accum_steps, -1, *x.shape[1:]), batch)
            carry_new, outputs = jax.lax.scan(computation, carry_init, batch)
            loss, ttt_stats = outputs
            loss = jnp.mean(loss)
            if output_ttt_stats:
                ttt_stats = tree_map(lambda x: jnp.mean(x, axis=0), ttt_stats)
            else:
                ttt_stats = None
            grads = jax.tree_util.tree_map(lambda x: x / FLAGS.accum_steps, carry_new["sum_grads"])


            train_state = train_state.apply_gradients(grads=grads)
            learning_rate = optimizer_info["learning_rate_schedule"](train_state.step)
            grads_norm = global_norm(grads)


            return (train_state, loss, ttt_stats, grads_norm, learning_rate, rng_generator())


    else:
        raise ValueError(f"Accum steps must >= 1, got {accum_steps}")


    return train_step



def make_minimal_forward_fn(model, model_config):
    """Create a minimal compiled forward function for zero-order training that handles caches."""
    def forward_fn(params, input_tokens, target_tokens, loss_masks, ttt_lr_mult, rng, cache):
        """Minimal forward pass that accepts and returns a cache."""
        variables = {'params': params}
        if cache is not None:
            # Add cache variables to the top level of the `variables` dict.
            for cache_type, cache_value in cache.items():
                variables[cache_type] = cache_value
        
        # Define which collections are mutable (i.e., will be updated and returned).
        mutable_collections = [
            'ttt_cache', 'conv_cache', 'pre_conv_cache', "mini_batch_counter",
            'ttt_cache_medium', 'ttt_cache_slow'
        ]
        
        model_outputs, updated_vars = model.apply(
            variables,
            input_tokens,
            ttt_lr_mult=ttt_lr_mult,
            deterministic=True,  # Use deterministic mode for ZO evaluation.
            rngs=rng,
            mutable=mutable_collections
        )
        
        loss, _ = cross_entropy_loss_and_accuracy(model_outputs.logits, target_tokens, loss_masks)
        
        # Return the loss and the dictionary of updated mutable states (the new cache).
        return loss, updated_vars
    return forward_fn


def compute_zero_order_gradient_estimate(compiled_forward_fn, params, chunked_batch, ttt_lr_mult, 
                                       rng_generator, num_chunks, num_perturbations, 
                                       perturbation_scale, verbose=False):
    """Compute zero-order gradient estimate for given number of chunks."""
    
    def compute_total_loss(params_to_eval):
        """Helper to compute total loss over all chunks for a given set of parameters."""
        total_loss = 0.0
        current_cache = None
        for i in range(num_chunks):
            if verbose:
                master_print(f"[ZO Debug] Processing chunk {i + 1}/{num_chunks}...")
            # Get the data for the current chunk.
            input_chunk = chunked_batch["input_tokens"][i]
            target_chunk = chunked_batch["target_tokens"][i]
            mask_chunk = chunked_batch["loss_masks"][i]
            
            # Each chunk gets a deterministic RNG for reproducibility.
            chunk_rng = rng_generator(["dropout", "params"])
            
            # Call the compiled forward function for one chunk.
            loss_chunk, updated_cache = compiled_forward_fn(
                params_to_eval, input_chunk, target_chunk, mask_chunk, ttt_lr_mult, chunk_rng, current_cache
            )
            
            total_loss += loss_chunk
            current_cache = updated_cache  # Pass the cache to the next chunk.
        
        return total_loss / num_chunks # Return the average loss over all chunks.

    # 1. Calculate baseline loss over all chunks.
    baseline_loss = compute_total_loss(params)
    
    # 2. Estimate gradient via perturbations.
    grad_estimate = tree_map(jnp.zeros_like, params)
    for pert_idx in range(num_perturbations):
        # Generate a random perturbation.
        perturbation_rng = rng_generator()
        perturbation = tree_map(
            lambda x: jax.random.normal(perturbation_rng, x.shape, dtype=x.dtype) * perturbation_scale,
            params
        )
        
        # Apply perturbation.
        perturbed_params = tree_map(lambda p, delta: p + delta, params, perturbation)
        
        # Calculate perturbed loss over all chunks.
        perturbed_loss = compute_total_loss(perturbed_params)
        
        # Calculate finite difference gradient.
        loss_diff = perturbed_loss - baseline_loss
        perturbation_grad = tree_map(
            lambda delta: (loss_diff / perturbation_scale) * delta, # Correct scaling
            perturbation
        )
        
        # Accumulate gradient estimate.
        grad_estimate = tree_map(
            lambda g, pg: g + pg / num_perturbations,
            grad_estimate, perturbation_grad
        )

    # Apply gradient clipping
    max_grad_norm = 1.0  
    grad_estimate = tree_map(
        lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm), 
        grad_estimate
    )
    
    return grad_estimate, baseline_loss


def make_debug_gradient_fn(model, model_config):
    """Create an uncompiled gradient function for debug purposes only."""
    def debug_gradient_fn(params, batch, ttt_lr_mult, rng):
        """Compute gradients using the uncompiled model for debug comparison."""
        rng_generator = JaxRNG(rng)
        
        def loss_fn(params):
            outputs = model.apply(
                {'params': params},
                batch["input_tokens"],
                ttt_lr_mult=ttt_lr_mult,
                deterministic=True,  # Use deterministic for debug consistency
                rngs=rng_generator(model_config.rng_keys()),
            )
            loss, _ = cross_entropy_loss_and_accuracy(
                outputs.logits, batch["target_tokens"], batch["loss_masks"]
            )
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        _, grads = grad_fn(params)
        return grads
    
    return debug_gradient_fn


def make_zero_order_train_step_fn(compiled_forward_fn, optimizer_info, FLAGS, model=None, model_config=None):
    """Create a PURE PYTHON zero-order training step that processes the batch in chunks."""
    
    # Create debug gradient function if debug mode is enabled
    debug_gradient_fn = None
    if FLAGS.zero_order_debug_cosine and model is not None and model_config is not None:
        debug_gradient_fn = make_debug_gradient_fn(model, model_config)
    
    def zero_order_train_step(train_state, rng, batch, ttt_lr_mult):
        """Pure Python zero-order training step that simulates a long sequence."""
        rng_generator = JaxRNG(rng)
        num_chunks = FLAGS.zero_order_num_chunks
        
        # Ensure the global batch size is divisible by the number of chunks.
        if batch["input_tokens"].shape[0] % num_chunks != 0:
            raise ValueError(
                f"Global batch size ({batch['input_tokens'].shape[0]}) must be divisible "
                f"by zero_order_num_chunks ({num_chunks})"
            )
        
        # Reshape the batch for chunking: (global_batch, seq_len) -> (num_chunks, per_chunk_batch, seq_len)
        chunked_batch = tree_map(
            lambda x: x.reshape(num_chunks, -1, *x.shape[1:]),
            batch
        )
        
        if FLAGS.zero_order_verbose:
            master_print("[ZO] Computing baseline loss over chunks...")
        
        # Compute main ZO gradient estimate
        grad_estimate, baseline_loss = compute_zero_order_gradient_estimate(
            compiled_forward_fn, train_state.params, chunked_batch, ttt_lr_mult,
            rng_generator, num_chunks, FLAGS.zero_order_num_perturbations, 
            FLAGS.zero_order_perturbation_scale, FLAGS.zero_order_verbose
        )
        
        if FLAGS.zero_order_verbose:
            master_print(f"[ZO] Baseline loss: {baseline_loss:.6f}")
        
        debug_metrics = {}
        
        # Debug: compute cosine similarities if enabled
        if FLAGS.zero_order_debug_cosine:
            if FLAGS.zero_order_verbose:
                master_print("[ZO Debug] Computing cosine similarities...")
            
            # Compute ZO gradient on single chunk (num_chunks=1)
            single_chunk_batch = tree_map(lambda x: x[:1], chunked_batch)  # Take only first chunk
            
            zo_single_grad, _ = compute_zero_order_gradient_estimate(
                compiled_forward_fn, train_state.params, single_chunk_batch, ttt_lr_mult,
                rng_generator, 1, FLAGS.zero_order_num_perturbations, 
                FLAGS.zero_order_perturbation_scale, verbose=False
            )
            
            # Compute standard Adam gradient using uncompiled debug function
            adam_grad = None
            if debug_gradient_fn is not None:
                try:
                    # Use single chunk in original format, but reduce batch size for memory safety
                    single_chunk_batch_orig = tree_map(lambda x: x[0], chunked_batch)
                    
                    # Reduce batch size for debug computation to avoid memory issues
                    debug_batch_size = min(4, single_chunk_batch_orig["input_tokens"].shape[0])
                    debug_batch = tree_map(lambda x: x[:debug_batch_size], single_chunk_batch_orig)
                    
                    # Compute Adam gradient using uncompiled function (no sharding constraints)
                    temp_rng = rng_generator()
                    adam_grad = debug_gradient_fn(train_state.params, debug_batch, ttt_lr_mult, temp_rng)
                    
                    if FLAGS.zero_order_verbose:
                        master_print(f"[ZO Debug] Computed Adam gradient from debug function (batch_size={debug_batch_size})")
                        
                except Exception as e:
                    if FLAGS.zero_order_verbose:
                        master_print(f"[ZO Debug] Adam gradient computation failed: {e}")
                    adam_grad = None
            
            # Compute cosine similarities
            cosine_zo_zo1 = compute_cosine_similarity(grad_estimate, zo_single_grad)
            debug_metrics["cosine_zo_zo1"] = float(cosine_zo_zo1)
            
            if adam_grad is not None:
                cosine_zo1_adam = compute_cosine_similarity(zo_single_grad, adam_grad)
                debug_metrics["cosine_zo1_adam"] = float(cosine_zo1_adam)
            
            if FLAGS.zero_order_verbose:
                master_print(f"[ZO Debug] Cosine(ZO_multi_chunk, ZO_single_chunk): {cosine_zo_zo1:.4f}")
                if adam_grad is not None:
                    master_print(f"[ZO Debug] Cosine(ZO_single_chunk, Adam): {cosine_zo1_adam:.4f}")
                else:
                    master_print("[ZO Debug] Adam gradient computation skipped (function not available or error)")

        # 3. Apply gradients (use main ZO estimate for actual updates).
        train_state = train_state.apply_gradients(grads=grad_estimate)
        
        # 4. Compute metrics.
        learning_rate = optimizer_info["learning_rate_schedule"](train_state.step)
        grads_norm = global_norm(grad_estimate)
        
        # TTT stats are not computed in ZO mode.
        return (train_state, baseline_loss, None, grads_norm, learning_rate, rng_generator(), debug_metrics)
    
    return zero_order_train_step



def should_use_zero_order(step: int, FLAGS) -> bool:
    """Determine whether to use zero-order training."""
    if not FLAGS.use_zero_order_training:
        return False
    if FLAGS.zero_order_frequency <= 0:
        return False
    if FLAGS.zero_order_frequency == 1:
        return True
    return (step % FLAGS.zero_order_frequency) == 0



def make_eval_step_fn(model, model_config):

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            {'params': train_state.params}, batch["input_tokens"], deterministic=True, rngs=rng_generator(model_config.rng_keys())
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
        metrics = dict(eval_loss=loss, eval_accuracy=accuracy)
        return rng_generator(), metrics


    return eval_step



def make_sharded_functions(model, optimizer, optimizer_info, model_config):

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)


    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            rng_generator(model_config.rng_keys()),
            input_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, FLAGS.seq_length), dtype=jnp.int32),
        )["params"]
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)


    # Create gradient training function
    gradient_train_step = make_train_step_fn(model, optimizer_info, model_config, FLAGS.accum_steps)

    # Create minimal forward function for zero-order (this will be compiled)
    minimal_forward_fn = None
    zero_order_train_step = None
    if FLAGS.use_zero_order_training:
        master_print("[COMPILE] Creating minimal forward function for zero-order training...")
        minimal_forward_fn = make_minimal_forward_fn(model, model_config)


    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(model_config.get_partition_rules(), train_state_shapes)
    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)


    sharded_init_fn = pjit(init_fn, in_shardings=PS(), out_shardings=train_state_partition)


    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )


    # Compile gradient training step
    master_print("[COMPILE] Compiling gradient training step...")
    sharded_gradient_train_step = pjit(
    gradient_train_step,
    in_shardings=(train_state_partition, PS(), PS(), PS()),  # 4 shardings
    out_shardings=(train_state_partition, PS(), PS(), PS(), PS(), PS()),
    static_argnums=(4,), # output_ttt_stats
    donate_argnums=(0,),
)
    master_print("[COMPILE] Gradient training step compiled!")


    # Compile ONLY the minimal forward function for zero-order.
    sharded_minimal_forward_fn = None
    if minimal_forward_fn is not None:
        master_print("[COMPILE] Compiling minimal forward function for zero-order (handles cache)...")
        sharded_minimal_forward_fn = pjit(
            minimal_forward_fn,
            # Shardings for: params, inputs, targets, mask, ttt_lr_mult, rng, cache
            in_shardings=(train_state_partition.params, PS(), PS(), PS(), PS(), PS(), PS()),
            # Shardings for: (loss, updated_cache)
            out_shardings=(PS(), PS()),
        )
        master_print("[COMPILE] Minimal forward function compiled!")
        # Create the pure Python ZO step, passing it the *compiled* forward function.
        master_print("[COMPILE] Creating zero-order training step (pure Python, no compilation)...")
        # Pass model and model_config for debug gradient computation if enabled
        zero_order_train_step = make_zero_order_train_step_fn(
            sharded_minimal_forward_fn, 
            optimizer_info, 
            FLAGS, 
            model if FLAGS.zero_order_debug_cosine else None,
            model_config if FLAGS.zero_order_debug_cosine else None
        )
        master_print("[COMPILE] Zero-order training step created!")
    else:
        master_print("[COMPILE] Zero-order training disabled, skipping forward function compilation")


    # Pure Python runtime selector (NO compilation)
    def runtime_train_step_selector(train_state, rng, batch, ttt_lr_mult, output_ttt_stats=False, use_zero_order=False):
        """Pure Python runtime selector - no compilation overhead."""
        if use_zero_order and zero_order_train_step is not None:
            # Call pure Python zero-order function directly.
            result = zero_order_train_step(train_state, rng, batch, ttt_lr_mult)
            # ZO function returns 7 values (including debug_metrics), standard returns 6
            if len(result) == 7:
                return result  # Return all 7 values including debug_metrics
            else:
                # Fallback: add empty debug_metrics if not returned
                return result + ({},)
        else:
            # Call compiled gradient training.
            result = sharded_gradient_train_step(train_state, rng, batch, ttt_lr_mult, output_ttt_stats)
            # Add empty debug_metrics to match ZO return signature
            return result + ({},)


    return (
        sharded_init_fn,
        sharded_create_trainstate_from_params,
        runtime_train_step_selector, # Pure Python function - NO pjit
        shard_fns,
        gather_fns,
        train_state_shapes,
        train_state_partition,
    )



def make_save_checkpoint(checkpointer, gather_fns, variant, flags_config_dict, model_config, global_batch_size):

    def save_checkpoint(train_state, train_loader, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(step=step, variant=variant, flags=flags_config_dict, model_config=model_config.to_dict())
        sampler_state_dict = {
            "random_state": train_loader.sampler.state_dict()["random_state"],
            "shuffle_log": train_loader.sampler.state_dict()["shuffle_log"],
            "counter": step * global_batch_size,
        }
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=deepcopy(sampler_state_dict),
            milestone=milestone,
            )


    return save_checkpoint



def make_get_ttt_lr_mult(model_config):

    if (
        hasattr(model_config, "ttt_base_lr_init")
        and model_config.ttt_base_lr_init > 0
        and model_config.ttt_base_lr_warmup > 0
    ):
        ttt_lr_mult_warmup_steps = model_config.ttt_base_lr_warmup
        ttt_lr_mult_init = model_config.ttt_base_lr_init
        ttt_lr_mult_peak = model_config.ttt_base_lr


        def get_ttt_lr_mult(step):
            ttt_lr_mult = ttt_lr_mult_init + min(1.0, (step - 1) / ttt_lr_mult_warmup_steps) * (
                ttt_lr_mult_peak - ttt_lr_mult_init
            )
            ttt_lr_mult = ttt_lr_mult / ttt_lr_mult_peak * jnp.ones((1,), dtype=jnp.bfloat16)
            return ttt_lr_mult


    else:
        def get_ttt_lr_mult(step):
            ttt_lr_mult = jnp.ones((1,), dtype=jnp.bfloat16)
            return ttt_lr_mult


    return get_ttt_lr_mult



def initialize_or_resume(
    checkpointer,
    train_loader,
    train_state_shapes,
    sharded_init_fn,
    shard_fns,
    sharded_create_trainstate_from_params,
    FLAGS,
):
    start_step = 1
    train_state, restored_params = None, None
    if FLAGS.resume_exp_name != "":
        assert FLAGS.load_part in ["trainstate", "trainstate_params"]
        master_print(f"Resuming from experiment: {FLAGS.resume_exp_name}...")
        ckpt_resume_dir = (
            FLAGS.load_part
            + "::"
            + osp.join(
                FLAGS.exp_dir,
                FLAGS.resume_exp_name,
                (
                    f"step_{int(FLAGS.resume_step)}/streaming_train_state_{int(FLAGS.resume_step)}"
                    if FLAGS.resume_step
                    else "streaming_train_state"
                ),
            )
        )
        train_state, restored_params = checkpointer.load_trainstate_checkpoint(
            ckpt_resume_dir, train_state_shapes, shard_fns
        )


        if FLAGS.load_part == "trainstate":
            start_step = int(jax.device_get(train_state.step)) + 1
            master_print(f"Resuming training from checkpoint at step {start_step - 1}...")
            dataset_pkl_filename = (
                f"step_{int(FLAGS.resume_step)}/dataset_{int(FLAGS.resume_step)}.pkl"
                if FLAGS.resume_step
                else "dataset.pkl"
            )
            dataset_resume_dir = osp.join(FLAGS.exp_dir, FLAGS.resume_exp_name, dataset_pkl_filename)
            train_loader.sampler.load_state_dict(deepcopy(mlxu.load_pickle(dataset_resume_dir)))


            if FLAGS.is_rollback_reshuffle:
                train_loader.sampler.is_rollback = True


    if train_state is None and restored_params is None:
        master_print("No checkpoint found, initializing from scratch...")
        train_state = sharded_init_fn(next_rng())
    elif train_state is None and restored_params is not None:
        master_print("Checkpoint found, initializing train state from params...")
        train_state = sharded_create_trainstate_from_params(restored_params)
        del restored_params


    return start_step, train_state, train_loader



def save_training_results(results_dict, exp_dir, exp_name):
    """Save training results to a text file."""
    results_file = osp.join(exp_dir, exp_name, "training_results.txt")
    with open(results_file, 'w') as f:
        f.write("# Training Results\n")
        f.write(f"experiment_name: {exp_name}\n")
        f.write(f"total_steps: {len(results_dict['steps'])}\n")
        f.write(f"final_loss: {results_dict['losses'][-1]:.6f}\n")
        f.write(f"min_loss: {min(results_dict['losses']):.6f}\n")
        f.write(f"final_lr: {results_dict['learning_rates'][-1]:.6e}\n")
        f.write(f"final_grad_norm: {results_dict['grad_norms'][-1]:.6f}\n")
        f.write("\n")
        f.write("# Detailed Metrics (step,loss,grad_norm,learning_rate,method)\n")
        for i in range(len(results_dict['steps'])):
            method = results_dict.get('methods', ['gradient'] * len(results_dict['steps']))[i]
            f.write(f"{results_dict['steps'][i]},{results_dict['losses'][i]:.6f},"
                f"{results_dict['grad_norms'][i]:.6f},{results_dict['learning_rates'][i]:.6e},{method}\n")



def count_model_parameters(params):
    """Count total and TTT-specific parameters."""
    flat_params = flatten_dict(params, sep='/')
    total_params = 0
    ttt_params = 0
    ttt_keywords = ['ttt_', 'W_1', 'W_2', 'b_1', 'b_2', 'ttt_norm', 'mini_batch_counter']
    param_breakdown = {}
    for param_name, param_value in flat_params.items():
        param_count = param_value.size
        total_params += param_count
        is_ttt_param = any(keyword in param_name.lower() for keyword in ttt_keywords)
        if is_ttt_param:
            ttt_params += param_count

        if 'embed' in param_name.lower():
            param_breakdown['embedding'] = param_breakdown.get('embedding', 0) + param_count
        elif any(kw in param_name.lower() for kw in ttt_keywords):
            param_breakdown['ttt'] = param_breakdown.get('ttt', 0) + param_count
        elif 'attention' in param_name.lower() or 'attn' in param_name.lower():
            param_breakdown['attention'] = param_breakdown.get('attention', 0) + param_count
        elif 'feed_forward' in param_name.lower() or 'ffn' in param_name.lower() or 'mlp' in param_name.lower():
            param_breakdown['feedforward'] = param_breakdown.get('feedforward', 0) + param_count
        elif 'norm' in param_name.lower():
            param_breakdown['normalization'] = param_breakdown.get('normalization', 0) + param_count
        elif 'lm_head' in param_name.lower() or 'output' in param_name.lower():
            param_breakdown['output'] = param_breakdown.get('output', 0) + param_count
        else:
            param_breakdown['other'] = param_breakdown.get('other', 0) + param_count
    return {
        'total_params': total_params,
        'ttt_params': ttt_params,
        'non_ttt_params': total_params - ttt_params,
        'breakdown': param_breakdown
    }



def print_model_parameter_info(params, model_config):
    """Print detailed parameter information."""
    param_info = count_model_parameters(params)
    master_print("="*60)
    master_print("MODEL PARAMETER INFORMATION")
    master_print("="*60)
    total_params = param_info['total_params']
    ttt_params = param_info['ttt_params']
    non_ttt_params = param_info['non_ttt_params']
    master_print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    master_print(f"TTT Parameters: {ttt_params:,} ({ttt_params/1e6:.2f}M)")
    master_print(f"Non-TTT Parameters: {non_ttt_params:,} ({non_ttt_params/1e6:.2f}M)")
    if total_params > 0:
        ttt_percentage = (ttt_params / total_params) * 100
        master_print(f"TTT Overhead: {ttt_percentage:.2f}% of total parameters")
    master_print("-" * 60)
    master_print("PARAMETER BREAKDOWN BY COMPONENT:")
    breakdown = param_info['breakdown']
    for component, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100 if total_params > 0 else 0
        master_print(f" {component.capitalize():15}: {count:,} ({count/1e6:.2f}M, {percentage:.1f}%)")
    master_print("-" * 60)
    master_print("MODEL CONFIGURATION:")
    master_print(f" Architecture: {getattr(model_config, 'seq_modeling_block', 'unknown')}")
    master_print(f" Hidden Size: {getattr(model_config, 'hidden_size', 'unknown'):,}")
    master_print(f" Number of Layers: {getattr(model_config, 'num_hidden_layers', 'unknown')}")
    master_print(f" Attention Heads: {getattr(model_config, 'num_attention_heads', 'unknown')}")
    master_print(f" Vocab Size: {getattr(model_config, 'vocab_size', 'unknown'):,}")
    master_print(f" Sequence Length: {getattr(model_config, 'max_sequence_length', 'unknown'):,}")
    if hasattr(model_config, 'mini_batch_size'):
        master_print(f" TTT Mini Batch Size: {model_config.mini_batch_size}")
    if hasattr(model_config, 'ttt_base_lr'):
        master_print(f" TTT Base LR: {model_config.ttt_base_lr}")
    master_print("="*60)



def print_zero_order_config(FLAGS):
    """Print zero-order training configuration."""
    if FLAGS.use_zero_order_training:
        master_print("="*60)
        master_print("ZERO-ORDER TRAINING CONFIGURATION")
        master_print("="*60)
        master_print(f"Enabled: {FLAGS.use_zero_order_training}")
        master_print(f"Frequency: {FLAGS.zero_order_frequency} (0=never, 1=always, N=every N steps)")
        master_print(f"Num Chunks: {FLAGS.zero_order_num_chunks} (effective sequence is longer)")
        master_print(f"Num Perturbations: {FLAGS.zero_order_num_perturbations}")
        master_print(f"Perturbation Scale: {FLAGS.zero_order_perturbation_scale}")
        master_print(f"Verbose Logging: {FLAGS.zero_order_verbose}")
        master_print(f"Debug Cosine Similarity: {FLAGS.zero_order_debug_cosine}")
        master_print("="*60)



def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)


    set_random_seed(FLAGS.seed)
    process_num = jax.process_count()
    global_dev_num = jax.device_count()
    local_dev_num = jax.local_device_count()
    master_process = jax.process_index() == 0


    dev_info = f"Process # {process_num}\tLocal dev # {local_dev_num}\tTotal dev # {global_dev_num}"
    master_print(dev_info)


    seq_length = FLAGS.seq_length
    global_batch_size = FLAGS.global_batch_size
    is_rollback_reshuffle = FLAGS.is_rollback_reshuffle


    # Print zero-order configuration
    if master_process:
        print_zero_order_config(FLAGS)


    # Create dataloader
    data_module = LMDataModule(
        dataset_name=FLAGS.dataset_name,
        dataset_config_name=None,
        tokenizer_name=FLAGS.tokenizer_name,
        cache_dir=FLAGS.dataset_path,
        max_length=seq_length,
        add_eos=True,
        batch_size=global_batch_size,
        batch_size_eval=global_batch_size,
        loader_workers=FLAGS.loader_workers,
        shuffle=True,
        fault_tolerant=True,
        drop_last=True,
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()


    # Model configuration
    if FLAGS.load_model_config != "":
        model_config = ModelConfig.load_config(FLAGS.load_model_config)
    else:
        raise RuntimeError(f"model_config must be specified")
    if FLAGS.update_model_config:
        update_dic = eval(FLAGS.update_model_config)
        print("update_dic", update_dic)
        for key, value in update_dic.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise KeyError(f"Update key {key} not in model_config")
    model_config.vocab_size = data_module.vocab_size
    model_config.max_sequence_length = seq_length
    # Ensure pad_token_id is properly set
    if not hasattr(model_config, 'pad_token_id') or model_config.pad_token_id is None:
        # Get pad_token_id from tokenizer
        tokenizer = data_module.tokenizer
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            model_config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            model_config.pad_token_id = tokenizer.eos_token_id
        else:
            model_config.pad_token_id = 0 # Fallback to 0
        master_print(f"Set model_config.pad_token_id = {model_config.pad_token_id}")
    flags_config_dict["model_config"] = model_config.to_dict()


    mesh = model_config.get_jax_mesh(FLAGS.mesh_dim)


    # Create WandB run and checkpointer
    if master_process:
        wandb_run = wandb.init(project="TTT-LM", config=flags_config_dict, name=FLAGS.exp_name)
        # Save W&B run ID for perplexity evaluation
        wandb_run_id = wandb_run.id
        wandb_id_file = osp.join(FLAGS.exp_dir, FLAGS.exp_name, "wandb_run_id.txt")
        os.makedirs(osp.dirname(wandb_id_file), exist_ok=True)
        with open(wandb_id_file, 'w') as f:
            f.write(wandb_run_id)
        master_print(f"W&B run ID saved: {wandb_run_id}")
        ckpt_dir = osp.join(FLAGS.exp_dir, FLAGS.exp_name)
        checkpointer = StreamingCheckpointer(FLAGS.checkpointer, ckpt_dir, enable=master_process)
    else:
        checkpointer = StreamingCheckpointer(FLAGS.checkpointer, None, enable=False)


    # Create model and optimizer
    model = CausalLM(model_config, dtype=get_float_dtype_by_name(FLAGS.dtype))
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer, get_weight_decay_mask(model_config.get_weight_decay_exclusions())
    )


    # TTT learning rate helper
    get_ttt_lr_mult = make_get_ttt_lr_mult(model_config)


    # Create sharded functions
    (
        sharded_init_fn,
        sharded_create_trainstate_from_params,
        runtime_train_step_selector, # Pure Python runtime selector
        shard_fns,
        gather_fns,
        train_state_shapes,
        train_state_partition,
    ) = make_sharded_functions(model, optimizer, optimizer_info, model_config)


    save_checkpoint = make_save_checkpoint(
        checkpointer, gather_fns, variant, flags_config_dict, model_config, global_batch_size
    )


    master_print(f"Using mesh: {mesh}")
    with mesh:
        sharded_rng = next_rng()


        start_step, train_state, train_loader = initialize_or_resume(
            checkpointer,
            train_loader,
            train_state_shapes,
            sharded_init_fn,
            shard_fns,
            sharded_create_trainstate_from_params,
            FLAGS,
        )


        # Print model parameter information
        if master_process:
            print_model_parameter_info(train_state.params, model_config)


        train_loader_iterator = iter(train_loader)


        # Initialize results tracking
        results_dict = {
            'steps': [],
            'losses': [],
            'grad_norms': [],
            'learning_rates': [],
            'methods': []
        }


        for step in tqdm(
            range(start_step, FLAGS.total_steps + 1),
            initial=start_step,
            total=FLAGS.total_steps,
            disable=not master_process,
            desc=f"Training {FLAGS.exp_name}",
        ):
            try:
                batch = next(train_loader_iterator)
            except StopIteration:
                train_loader.sampler.counter = 0
                train_loader_iterator = iter(train_loader)
                batch = next(train_loader_iterator)


            if is_rollback_reshuffle:
                sampler_state_dict = {
                    "random_state": train_loader.sampler.state_dict()["random_state"],
                    "shuffle_log": train_loader.sampler.state_dict()["shuffle_log"],
                    "counter": (step - 1) * global_batch_size,
                }
                if master_process and FLAGS.resume_exp_name != "":
                    master_print("Updating sampler state after rollback...")
                    dataset_pkl_filename = (
                        f"step_{int(FLAGS.resume_step)}/dataset_{int(FLAGS.resume_step)}.pkl"
                        if FLAGS.resume_step
                        else "dataset_state.pkl"
                    )
                    dataset_resume_dir = osp.join(FLAGS.exp_dir, FLAGS.resume_exp_name, dataset_pkl_filename)
                    mlxu.save_pickle(deepcopy(sampler_state_dict), dataset_resume_dir)
                    is_rollback_reshuffle = False
                    master_print("Finished updating sampler state.")


            for k in batch.keys():
                batch[k] = batch[k].numpy()


            ttt_lr_mult = get_ttt_lr_mult(step)
            # Determine training method
            use_zero_order = should_use_zero_order(step, FLAGS)
            method = "zero-order" if use_zero_order else "gradient"
            
            output_ttt_stats = (
                FLAGS.save_milestone_freq > 0
                and step % FLAGS.save_milestone_freq == 0
                and model_config.seq_modeling_block != "self_attention"
                and not use_zero_order # TTT stats not supported in zero-order mode
            )


            train_state, loss, ttt_stats, grads_norm, learning_rate, sharded_rng, debug_metrics = runtime_train_step_selector(
                train_state, sharded_rng, batch, ttt_lr_mult, output_ttt_stats, use_zero_order
            )


            # Store results
            if master_process:
                results_dict['steps'].append(step)
                results_dict['losses'].append(float(loss.item()))
                results_dict['grad_norms'].append(float(grads_norm.item()))
                results_dict['learning_rates'].append(float(learning_rate.item()))
                results_dict['methods'].append(method)


                # Convert training method to numeric for WandB compatibility
                training_method_numeric = 1 if use_zero_order else 0
                
                log_dict = {
                    "Train Loss": loss.item(),
                    "Gradient Norm": grads_norm.item(),
                    "Learning Rate": learning_rate.item(),
                    "Training Method (0=grad, 1=ZO)": training_method_numeric,
                }
                
                # Add debug metrics if available
                if debug_metrics:
                    for key, value in debug_metrics.items():
                        log_dict[f"Debug/{key}"] = value
                
                wandb.log(log_dict, step=step)


                if output_ttt_stats and ttt_stats is not None:
                    for layer in range(len(ttt_stats)):
                        ttt_stats_layer = process_allgather(ttt_stats[layer])
                        n_mini_batch = len(ttt_stats_layer[0])
                        x_axis = [model_config.mini_batch_size * i for i in range(1, n_mini_batch + 1)]
                        log_ttt_stats(layer, ttt_stats_layer, x_axis, step)


                if (FLAGS.save_checkpoint_freq > 0 and step % FLAGS.save_checkpoint_freq == 0) or (
                    step == FLAGS.total_steps
                ):
                    master_print(f"Saving checkpoint at step {step}, do not kill...")
                    save_checkpoint(train_state, train_loader, (FLAGS.save_milestone_freq > 0 and step % FLAGS.save_milestone_freq == 0))


        if step == FLAGS.total_steps:
            master_print("Training has completed!")
            if master_process:
                master_print("Saving training results...")
                save_training_results(results_dict, FLAGS.exp_dir, FLAGS.exp_name)
                master_print(f"Results saved to {osp.join(FLAGS.exp_dir, FLAGS.exp_name, 'training_results.txt')}")


def log_perplexity_to_wandb(exp_dir, exp_name, perplexity_results):
    """Resume W&B run and log perplexity results."""
    wandb_id_file = osp.join(exp_dir, exp_name, "wandb_run_id.txt")
    if not osp.exists(wandb_id_file):
        master_print("W&B run ID file not found, cannot log perplexity results")
        return
    
    with open(wandb_id_file, 'r') as f:
        wandb_run_id = f.read().strip()
    
    # Resume the existing W&B run
    wandb.init(project="TTT-LM", id=wandb_run_id, resume="must")
    
    # Log perplexity results
    if isinstance(perplexity_results, dict):
        # If results contain multiple metrics
        log_dict = {}
        for key, value in perplexity_results.items():
            if key.lower().startswith('ppl') or 'perplexity' in key.lower():
                log_dict[f"Final_{key}"] = value
            else:
                log_dict[f"Final_{key}"] = value
        wandb.log(log_dict)
    else:
        # If results is a single perplexity value
        wandb.log({"Final_Perplexity": perplexity_results})
    
    master_print(f"Perplexity results logged to W&B run: {wandb_run_id}")
    wandb.finish()


if __name__ == "__main__":
    mlxu.run(main)