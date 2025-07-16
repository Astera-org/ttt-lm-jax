#!/usr/bin/env python3
"""
TTT Language Model Perplexity Evaluation with Weight Norm Tracking

This script evaluates perplexity and tracks TTT weight norms across multiple books,
with optimized chunking for efficient computation and comprehensive debugging.
"""

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils, pjit
from transformers import AutoTokenizer
import optax
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import random
import mlxu
from flax.core import freeze
from typing import List, Tuple, Dict, Optional, Any
import traceback
import wandb

from ttt.models.model import ModelConfig, CausalLM, CONFIGS
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.infra.jax_utils import (
    match_partition_rules,
    make_shard_and_gather_fns,
    next_rng as global_next_rng,
    set_random_seed,
    JaxRNG,
    JaxDistributedConfig,
    get_float_dtype_by_name
)

# Configuration
FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=0,
    mesh_dim="1,-1,1",
    dtype="fp32",
    load_model_config="",
    update_model_config="",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    checkpointer=StreamingCheckpointer.get_default_config(),
    exp_dir="",
    exp_name="",
    jax_distributed=JaxDistributedConfig.get_default_config(),
    compute_chunk_size=8192,
    books_dir="./data/gutenberg/",
    num_books=30,
    tokens_per_book=8192,
    ppl_seq_size=2048,
    skip_start_chars=1024,
    debug_mode=False,
    # W&B integration flags
    use_wandb=True,
    wandb_project="TTT-LM",
    log_to_wandb=True,
)

class TTTNormCalculator:
    """Handles TTT weight norm calculations with comprehensive debugging."""
    
    @staticmethod
    def calculate_norm(ttt_cache: Any, debug: bool = False) -> Optional[float]:
        """
        Calculate L2 norm of TTT weights from cache.
        
        Args:
            ttt_cache: TTT cache containing weight updates (often FrozenDict)
            debug: Enable detailed debugging output
            
        Returns:
            L2 norm as float or None if calculation fails
        """
        from flax.core.frozen_dict import FrozenDict
        
        if debug:
            print(f"[TTT_NORM] Cache type: {type(ttt_cache)}")
            print(f"[TTT_NORM] Cache is None: {ttt_cache is None}")
            print(f"[TTT_NORM] Is FrozenDict: {isinstance(ttt_cache, FrozenDict)}")
        
        if ttt_cache is None:
            if debug:
                print("[TTT_NORM] Cache is None, returning None")
            return None
        
        try:
            total_norm_squared = 0.0
            total_elements = 0
            arrays_found = 0
            
            def traverse_cache(tree: Any, depth: int = 0, path: str = "") -> None:
                nonlocal total_norm_squared, total_elements, arrays_found
                
                if debug and depth < 4:  # Increased depth for FrozenDict
                    indent = "  " * depth
                    print(f"[TTT_NORM] {indent}Traversing {path}: {type(tree).__name__}")
                
                # Handle FrozenDict specifically
                if isinstance(tree, FrozenDict):
                    if debug and depth < 3:
                        print(f"[TTT_NORM] {indent}FrozenDict keys: {list(tree.keys())}")
                    # FrozenDict behaves like dict but use .items() explicitly
                    for key, value in tree.items():
                        traverse_cache(value, depth + 1, f"{path}/{key}")
                        
                elif isinstance(tree, dict):
                    if debug and depth < 3:
                        print(f"[TTT_NORM] {indent}Dict keys: {list(tree.keys())}")
                    for key, value in tree.items():
                        traverse_cache(value, depth + 1, f"{path}/{key}")
                        
                elif isinstance(tree, (list, tuple)):
                    if debug and depth < 3:
                        print(f"[TTT_NORM] {indent}Sequence length: {len(tree)}")
                    for i, item in enumerate(tree):
                        traverse_cache(item, depth + 1, f"{path}[{i}]")
                        
                elif hasattr(tree, 'shape') and hasattr(tree, 'dtype'):
                    # This is likely a JAX/numpy array
                    arrays_found += 1
                    if debug:
                        print(f"[TTT_NORM] {indent}Array at {path}: shape={tree.shape}, dtype={tree.dtype}")
                        
                        # Print some array statistics
                        array_min = float(jnp.min(tree))
                        array_max = float(jnp.max(tree))
                        array_mean = float(jnp.mean(tree))
                        print(f"[TTT_NORM] {indent}  Stats: min={array_min:.6f}, max={array_max:.6f}, mean={array_mean:.6f}")
                    
                    # Calculate squared norm
                    try:
                        squared_norm = jnp.sum(jnp.square(tree))
                        total_norm_squared += squared_norm
                        total_elements += tree.size
                        
                        if debug:
                            squared_norm_val = float(jax.device_get(squared_norm))
                            print(f"[TTT_NORM] {indent}  Added {tree.size} elements, squared_norm={squared_norm_val:.6f}")
                    except Exception as e:
                        if debug:
                            print(f"[TTT_NORM] {indent}  Error processing array: {e}")
                
                elif debug and depth < 3:
                    print(f"[TTT_NORM] {indent}Other type: {type(tree)} = {str(tree)[:100]}")
            
            # Traverse the cache structure
            traverse_cache(ttt_cache)
            
            if debug:
                print(f"[TTT_NORM] Summary: {arrays_found} arrays, {total_elements} elements")
                total_norm_squared_val = float(jax.device_get(total_norm_squared)) if hasattr(total_norm_squared, 'device_get') else float(total_norm_squared)
                print(f"[TTT_NORM] Total squared norm: {total_norm_squared_val:.6f}")
            
            if total_elements > 0:
                norm = jnp.sqrt(total_norm_squared)
                norm_value = float(jax.device_get(norm))
                if debug:
                    print(f"[TTT_NORM] Final norm: {norm_value:.6f}")
                return norm_value
            else:
                if debug:
                    print("[TTT_NORM] No elements found")
                return None
                
        except Exception as e:
            print(f"[TTT_NORM] ERROR in norm calculation: {e}")
            if debug:
                traceback.print_exc()
            return None
    
    @staticmethod
    def inspect_frozen_dict_structure(frozen_dict: Any, max_depth: int = 5) -> None:
        """
        Detailed inspection of FrozenDict structure for debugging.
        
        Args:
            frozen_dict: The FrozenDict to inspect
            max_depth: Maximum depth to traverse
        """
        from flax.core.frozen_dict import FrozenDict
        
        print(f"\n[TTT_INSPECT] Detailed FrozenDict Structure Analysis")
        print(f"[TTT_INSPECT] Type: {type(frozen_dict)}")
        print(f"[TTT_INSPECT] Is FrozenDict: {isinstance(frozen_dict, FrozenDict)}")
        
        if not isinstance(frozen_dict, FrozenDict):
            print(f"[TTT_INSPECT] Not a FrozenDict, cannot inspect")
            return
        
        def inspect_recursive(obj: Any, depth: int = 0, path: str = "root") -> None:
            if depth > max_depth:
                print(f"{'  ' * depth}[Max depth reached]")
                return
                
            indent = "  " * depth
            
            if isinstance(obj, FrozenDict):
                print(f"{indent}{path}: FrozenDict with {len(obj)} keys")
                for key in obj.keys():
                    print(f"{indent}  Key: {key} -> {type(obj[key])}")
                    if hasattr(obj[key], 'shape'):
                        print(f"{indent}    Shape: {obj[key].shape}, dtype: {obj[key].dtype}")
                    inspect_recursive(obj[key], depth + 1, f"{path}.{key}")
                    
            elif isinstance(obj, dict):
                print(f"{indent}{path}: Dict with {len(obj)} keys")
                for key in obj.keys():
                    print(f"{indent}  Key: {key} -> {type(obj[key])}")
                    inspect_recursive(obj[key], depth + 1, f"{path}.{key}")
                    
            elif isinstance(obj, (list, tuple)):
                print(f"{indent}{path}: {type(obj).__name__} with {len(obj)} items")
                for i, item in enumerate(obj[:3]):  # Only show first 3 items
                    print(f"{indent}  [{i}]: {type(item)}")
                    inspect_recursive(item, depth + 1, f"{path}[{i}]")
                if len(obj) > 3:
                    print(f"{indent}  ... and {len(obj) - 3} more items")
                    
            elif hasattr(obj, 'shape'):
                stats = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'size': obj.size,
                    'min': float(jnp.min(obj)),
                    'max': float(jnp.max(obj)),
                    'mean': float(jnp.mean(obj)),
                    'std': float(jnp.std(obj))
                }
                print(f"{indent}{path}: Array {stats}")
            else:
                print(f"{indent}{path}: {type(obj)} = {str(obj)[:50]}")
        
        inspect_recursive(frozen_dict)

class BookLoader:
    """Handles loading and tokenizing books for evaluation."""
    
    @staticmethod
    def load_books(books_dir: str, tokenizer: Any, num_books: int, 
                   tokens_per_book: int, skip_start_chars: int = 1024) -> List[Tuple[str, str]]:
        """
        Load books that meet the token requirements.
        
        Args:
            books_dir: Directory containing book text files
            tokenizer: Tokenizer for counting tokens
            num_books: Number of books to load
            tokens_per_book: Minimum tokens required per book
            skip_start_chars: Characters to skip at beginning of each book
            
        Returns:
            List of (book_id, book_text) tuples
        """
        print(f"[BOOKS] Loading books from {books_dir}")
        book_files = list(glob.glob(os.path.join(books_dir, "*.txt")))
        
        if not book_files:
            print(f"[BOOKS] ERROR: No book files found in {books_dir}")
            return []
        
        print(f"[BOOKS] Found {len(book_files)} book files")
        random.shuffle(book_files)
        
        books = []
        attempted = 0
        skipped = 0
        
        for filepath in book_files:
            if len(books) >= num_books:
                break
                
            book_id = Path(filepath).stem
            attempted += 1
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    book_text = f.read()
                
                # Skip beginning characters (often metadata)
                book_text = book_text[skip_start_chars:]
                
                # Quick length check before tokenization
                if len(book_text) < tokens_per_book * 4:  # Rough estimate
                    print(f"[BOOKS] Skipping {book_id}: too short (estimated)")
                    skipped += 1
                    continue
                
                # Tokenize to get exact count
                tokenized = tokenizer(
                    book_text, 
                    return_tensors="np", 
                    return_attention_mask=False,
                    max_length=tokens_per_book,
                    truncation=True
                )
                token_count = tokenized.input_ids.shape[1]
                
                if token_count >= tokens_per_book:
                    books.append((book_id, book_text))
                    print(f"[BOOKS] Loaded {book_id}: {token_count} tokens (using {tokens_per_book})")
                else:
                    print(f"[BOOKS] Skipping {book_id}: {token_count} tokens (need {tokens_per_book})")
                    skipped += 1
                    
            except Exception as e:
                print(f"[BOOKS] Error loading {filepath}: {e}")
                skipped += 1
        
        print(f"[BOOKS] Summary: {len(books)} loaded, {skipped} skipped, {attempted} attempted")
        return books

class PerplexityCalculator:
    """Handles perplexity calculation with TTT norm tracking."""
    
    def __init__(self, model: Any, model_config: Any, compiled_fn: Any, 
                 norm_calculator: TTTNormCalculator, debug: bool = False):
        self.model = model
        self.model_config = model_config
        self.compiled_fn = compiled_fn
        self.norm_calculator = norm_calculator
        self.debug = debug
    
    def calculate_book_perplexity(self, params: Any, input_ids: jnp.ndarray, 
                                book_id: str, tokens_per_book: int, 
                                ppl_seq_size: int, compute_chunk_size: int) -> Dict[str, Any]:
        """
        Calculate perplexity and TTT norms for a single book.
        
        Returns:
            Dictionary with results including positions, perplexities, and norms
        """
        print(f"[CALC] Starting calculation for book {book_id}")
        start_time = time.time()
        
        # Initialize tracking lists
        seq_ending_positions = []
        seq_perplexities = []
        seq_ttt_norms = []
        total_tokens = 0
        
        # Initialize cache
        current_cache = None
        
        # Ensure we have batch dimension
        if input_ids.ndim == 1:
            input_ids = jnp.expand_dims(input_ids, axis=0)
        
        # Limit to tokens_per_book
        sequence_length = min(input_ids.shape[1], tokens_per_book)
        input_ids = input_ids[:, :sequence_length]
        
        print(f"[CALC] Processing {sequence_length} tokens in chunks of {compute_chunk_size}")
        
        chunk_count = 0
        for chunk_start in range(0, sequence_length, compute_chunk_size):
            chunk_end = min(chunk_start + compute_chunk_size, sequence_length)
            chunk = input_ids[:, chunk_start:chunk_end]
            original_chunk_len = chunk.shape[1]
            chunk_count += 1
            
            if original_chunk_len <= 1:
                print(f"[CALC] Skipping chunk {chunk_count}: too short ({original_chunk_len})")
                continue
            
            print(f"[CALC] Chunk {chunk_count}: {chunk_start}:{chunk_end} ({original_chunk_len} tokens)")
            
            # Pad chunk if necessary
            padded_chunk = self._pad_chunk(chunk, original_chunk_len)
            
            # Forward pass
            try:
                # Generate PRNG keys outside the compiled function
                rng = JaxRNG(global_next_rng())
                model_rngs = rng(self.model.config.rng_keys())
                
                model_outputs, updated_vars = self.compiled_fn(params, padded_chunk, current_cache, model_rngs)
                
                # Extract TTT norm
                ttt_norm = None
                if 'ttt_cache' in updated_vars:
                    ttt_norm = self.norm_calculator.calculate_norm(
                        updated_vars['ttt_cache'], 
                        debug=self.debug and chunk_count <= 2
                    )
                
                # Update cache for next iteration
                current_cache = self._update_cache(updated_vars)
                
                # Calculate perplexities for subsequences within this chunk
                chunk_results = self._calculate_chunk_perplexities(
                    model_outputs.logits[:, :original_chunk_len],
                    chunk[:, :original_chunk_len],
                    chunk_start,
                    ppl_seq_size,
                    ttt_norm
                )
                
                # Accumulate results
                seq_ending_positions.extend(chunk_results['positions'])
                seq_perplexities.extend(chunk_results['perplexities'])
                seq_ttt_norms.extend(chunk_results['norms'])
                total_tokens += chunk_results['tokens']
                
                if self.debug:
                    print(f"[CALC] Chunk {chunk_count}: {len(chunk_results['positions'])} sequences, TTT norm: {ttt_norm}")
                
            except Exception as e:
                print(f"[CALC] ERROR in chunk {chunk_count}: {e}")
                if self.debug:
                    traceback.print_exc()
                continue
        
        calc_time = time.time() - start_time
        print(f"[CALC] Book {book_id} completed: {len(seq_ending_positions)} sequences, "
              f"{total_tokens} tokens, {calc_time:.2f}s")
        
        return {
            'book_id': book_id,
            'seq_ending_positions': seq_ending_positions,
            'seq_perplexities': seq_perplexities,
            'seq_ttt_norms': seq_ttt_norms,
            'total_tokens': total_tokens,
            'calc_time': calc_time
        }
    
    def _pad_chunk(self, chunk: jnp.ndarray, original_len: int) -> jnp.ndarray:
        """Pad chunk to model's expected sequence length."""
        max_len = self.model_config.max_sequence_length
        if original_len < max_len:
            padding_len = max_len - original_len
            padding = jnp.full((1, padding_len), self.model_config.pad_token_id, dtype=chunk.dtype)
            return jnp.concatenate([chunk, padding], axis=1)
        return chunk
    
    def _update_cache(self, updated_vars: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract cache variables for next iteration."""
        cache_types = ['ttt_cache', 'pre_conv_cache', 'conv_cache',
        "mini_batch_counter",
        "ttt_cache_slow","ttt_cache_medium"]
        cache = {}
        
        for cache_type in cache_types:
            if cache_type in updated_vars:
                cache[cache_type] = updated_vars[cache_type]
        
        return cache if cache else None
    
    def _calculate_chunk_perplexities(self, logits: jnp.ndarray, tokens: jnp.ndarray,
                                    chunk_start: int, ppl_seq_size: int, 
                                    ttt_norm: Optional[float]) -> Dict[str, Any]:
        """Calculate perplexities for subsequences within a chunk."""
        positions = []
        perplexities = []
        norms = []
        total_tokens = 0
        
        original_len = logits.shape[1]
        
        for subseq_start in range(0, original_len, ppl_seq_size):
            subseq_end = min(subseq_start + ppl_seq_size, original_len)
            subseq_len = subseq_end - subseq_start
            
            if subseq_len <= 1:
                continue
            
            # Extract subsequence
            subseq_logits = logits[:, subseq_start:subseq_end]
            subseq_tokens = tokens[:, subseq_start:subseq_end]
            
            # Calculate perplexity (predict next token)
            shift_logits = subseq_logits[:, :-1]
            shift_labels = subseq_tokens[:, 1:]
            
            if shift_labels.size == 0:
                continue
            
            # Cross entropy calculation
            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            one_hot_labels = jax.nn.one_hot(shift_labels, self.model_config.vocab_size)
            cross_entropy = -jnp.sum(one_hot_labels * log_probs, axis=-1)
            
            # Perplexity
            ce_sum = jnp.sum(cross_entropy)
            token_count = shift_labels.size
            
            if token_count > 0:
                perplexity = jnp.exp(ce_sum / token_count)
                perplexity_val = float(jax.device_get(perplexity))
                
                # Record results
                absolute_end_pos = chunk_start + subseq_end
                positions.append(absolute_end_pos)
                perplexities.append(perplexity_val)
                norms.append(ttt_norm)
                total_tokens += token_count
        
        return {
            'positions': positions,
            'perplexities': perplexities,
            'norms': norms,
            'tokens': total_tokens
        }

class ResultsVisualizer:
    """Handles plotting and visualization of results."""
    
    @staticmethod
    def plot_results(book_results: List[Dict[str, Any]], exp_name: str, 
                    exp_dir: str, ppl_seq_size: int, compute_chunk_size: int, 
                    tokens_per_book: int, wandb_run: Optional[Any] = None) -> None:
        """Create and save perplexity and TTT norm plots."""
        print("[VIZ] Creating visualizations...")
        
        # Plot perplexity
        ppl_fig = ResultsVisualizer._plot_perplexity(
            book_results, exp_name, exp_dir, ppl_seq_size, 
            compute_chunk_size, tokens_per_book
        )
        
        # Plot TTT norms
        norm_fig = ResultsVisualizer._plot_ttt_norms(
            book_results, exp_name, exp_dir, ppl_seq_size, 
            compute_chunk_size, tokens_per_book
        )
        
        # Log plots to W&B
        if wandb_run is not None:
            try:
                # Log as images
                if ppl_fig is not None:
                    wandb_run.log({"perplexity_evaluation/perplexity_plot": wandb.Image(ppl_fig)})
                if norm_fig is not None:
                    wandb_run.log({"perplexity_evaluation/ttt_norm_plot": wandb.Image(norm_fig)})
                
                # Log as interactive plots
                ResultsVisualizer._log_interactive_plots_to_wandb(wandb_run, book_results)
                
                print("[VIZ] Plots logged to W&B")
            except Exception as e:
                print(f"[VIZ] Error logging plots to W&B: {e}")
        
        # Print statistics
        ResultsVisualizer._print_statistics(book_results)
    
    @staticmethod
    def _log_interactive_plots_to_wandb(wandb_run: Any, book_results: List[Dict[str, Any]]) -> None:
        """Log interactive plots directly to W&B."""
        try:
            print("[VIZ] Creating interactive W&B plots...")
            
            # Prepare data for plotting
            all_data = []
            position_data = {}
            norm_position_data = {}
            
            for result in book_results:
                book_id = result['book_id']
                positions = result['seq_ending_positions']
                perplexities = result['seq_perplexities']
                norms = result['seq_ttt_norms']
                
                # Collect data for individual book lines
                for pos, ppl, norm in zip(positions, perplexities, norms):
                    all_data.append({
                        'book_id': book_id,
                        'position': pos,
                        'perplexity': ppl,
                        'ttt_norm': norm if norm is not None else None
                    })
                    
                    # Collect for median calculation
                    if pos not in position_data:
                        position_data[pos] = []
                    position_data[pos].append(ppl)
                    
                    if norm is not None:
                        if pos not in norm_position_data:
                            norm_position_data[pos] = []
                        norm_position_data[pos].append(norm)
            
            # Create perplexity plot data
            perplexity_data = []
            
            # Add individual book data
            for data_point in all_data:
                perplexity_data.append([
                    data_point['position'],
                    data_point['perplexity'],
                    data_point['book_id'],
                    'individual'
                ])
            
            # Add median line data
            sorted_positions = sorted(position_data.keys())
            for pos in sorted_positions:
                median_ppl = np.median(position_data[pos])
                perplexity_data.append([
                    pos,
                    median_ppl,
                    'median',
                    'median'
                ])
            
            # Log perplexity plot
            perplexity_table = wandb.Table(
                data=perplexity_data,
                columns=['position', 'perplexity', 'book_id', 'type']
            )
            
            wandb_run.log({
                "perplexity_evaluation/interactive_perplexity": wandb.plot.line(
                    perplexity_table,
                    x='position',
                    y='perplexity',
                    color='book_id',
                    title='Perplexity Progression (Interactive)'
                )
            })
            
            # Create separate median-only perplexity plot
            median_ppl_data = []
            for pos in sorted_positions:
                median_ppl = np.median(position_data[pos])
                median_ppl_data.append([pos, median_ppl])
            
            if median_ppl_data:
                median_ppl_table = wandb.Table(
                    data=median_ppl_data,
                    columns=['position', 'median_perplexity']
                )
                
                wandb_run.log({
                    "perplexity_evaluation/median_perplexity": wandb.plot.line(
                        median_ppl_table,
                        x='position',
                        y='median_perplexity',
                        title='Median Perplexity Progression'
                    )
                })
            
            # Create TTT norm plot data if available
            if norm_position_data:
                norm_data = []
                
                # Add individual book norm data
                for data_point in all_data:
                    if data_point['ttt_norm'] is not None:
                        norm_data.append([
                            data_point['position'],
                            data_point['ttt_norm'],
                            data_point['book_id'],
                            'individual'
                        ])
                
                # Add median norm line data
                sorted_norm_positions = sorted(norm_position_data.keys())
                for pos in sorted_norm_positions:
                    median_norm = np.median(norm_position_data[pos])
                    norm_data.append([
                        pos,
                        median_norm,
                        'median',
                        'median'
                    ])
                
                # Log TTT norm plot
                norm_table = wandb.Table(
                    data=norm_data,
                    columns=['position', 'ttt_norm', 'book_id', 'type']
                )
                
                wandb_run.log({
                    "perplexity_evaluation/interactive_ttt_norm": wandb.plot.line(
                        norm_table,
                        x='position',
                        y='ttt_norm',
                        color='book_id',
                        title='TTT Weight Norm Progression (Interactive)'
                    )
                })
                
                # Create separate median-only TTT norm plot
                median_norm_data = []
                for pos in sorted_norm_positions:
                    median_norm = np.median(norm_position_data[pos])
                    median_norm_data.append([pos, median_norm])
                
                if median_norm_data:
                    median_norm_table = wandb.Table(
                        data=median_norm_data,
                        columns=['position', 'median_ttt_norm']
                    )
                    
                    wandb_run.log({
                        "perplexity_evaluation/median_ttt_norm": wandb.plot.line(
                            median_norm_table,
                            x='position',
                            y='median_ttt_norm',
                            title='Median TTT Weight Norm Progression'
                        )
                    })
                
                # Also log TTT norm as a scatter plot for better visualization
                ttt_scatter_data = []
                for data_point in all_data:
                    if data_point['ttt_norm'] is not None:
                        ttt_scatter_data.append([
                            data_point['position'],
                            data_point['ttt_norm'],
                            data_point['book_id']
                        ])
                
                if ttt_scatter_data:
                    ttt_scatter_table = wandb.Table(
                        data=ttt_scatter_data,
                        columns=['position', 'ttt_norm', 'book_id']
                    )
                    
                    wandb_run.log({
                        "perplexity_evaluation/ttt_norm_scatter": wandb.plot.scatter(
                            ttt_scatter_table,
                            x='position',
                            y='ttt_norm',
                            title='TTT Weight Norm vs Position (Scatter)'
                        )
                    })
                
                # Create histogram of TTT norm values
                ttt_hist_data = [[data_point['ttt_norm']] for data_point in all_data if data_point['ttt_norm'] is not None]
                if ttt_hist_data:
                    ttt_hist_table = wandb.Table(
                        data=ttt_hist_data,
                        columns=['ttt_norm']
                    )
                    
                    wandb_run.log({
                        "perplexity_evaluation/ttt_norm_histogram": wandb.plot.histogram(
                            ttt_hist_table,
                            value='ttt_norm',
                            title='TTT Weight Norm Distribution'
                        )
                    })
            
            # Create scatter plot for perplexity vs position
            scatter_data = []
            for data_point in all_data:
                scatter_data.append([
                    data_point['position'],
                    data_point['perplexity'],
                    data_point['book_id']
                ])
            
            scatter_table = wandb.Table(
                data=scatter_data,
                columns=['position', 'perplexity', 'book_id']
            )
            
            wandb_run.log({
                "perplexity_evaluation/perplexity_scatter": wandb.plot.scatter(
                    scatter_table,
                    x='position',
                    y='perplexity',
                    title='Perplexity vs Position (Scatter)'
                )
            })
            
            # Create histogram of perplexity values
            hist_data = [[data_point['perplexity']] for data_point in all_data]
            hist_table = wandb.Table(
                data=hist_data,
                columns=['perplexity']
            )
            
            wandb_run.log({
                "perplexity_evaluation/perplexity_histogram": wandb.plot.histogram(
                    hist_table,
                    value='perplexity',
                    title='Perplexity Distribution'
                )
            })
            
            print("[VIZ] Interactive plots logged to W&B successfully")
            
        except Exception as e:
            print(f"[VIZ] Error creating interactive W&B plots: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _plot_perplexity(book_results: List[Dict[str, Any]], exp_name: str, 
                        exp_dir: str, ppl_seq_size: int, compute_chunk_size: int, 
                        tokens_per_book: int) -> Optional[plt.Figure]:
        """Plot perplexity progression."""
        fig = plt.figure(figsize=(14, 8))
        
        all_perplexities = []
        position_data = {}
        
        # Plot individual books (light)
        for result in book_results:
            positions = result['seq_ending_positions']
            perplexities = result['seq_perplexities']
            
            all_perplexities.extend(perplexities)
            
            # Collect for median calculation
            for pos, ppl in zip(positions, perplexities):
                if pos not in position_data:
                    position_data[pos] = []
                position_data[pos].append(ppl)
            
            # Plot book line
            plt.plot(positions, perplexities, linewidth=1.0, alpha=0.1, 
                    marker='+', markersize=2, label=f'Book {result["book_id"]}')
        
        # Plot median line
        if position_data:
            sorted_positions = sorted(position_data.keys())
            median_perplexities = [np.median(position_data[pos]) for pos in sorted_positions]
            
            plt.plot(sorted_positions, median_perplexities, linewidth=2, alpha=1.0,
                    color='red', marker='o', markersize=4, 
                    label='Median across all books', zorder=10)
            
            # Reference line at first median
            if median_perplexities:
                plt.axhline(y=median_perplexities[0], color='blue', linestyle='--',
                           linewidth=1, alpha=0.7, 
                           label=f'First value ({median_perplexities[0]:.2f})')
        
        # Format plot
        plt.title(f'Perplexity Progression ({exp_name})\n'
                 f'{ppl_seq_size} tokens/seq, {compute_chunk_size} chunk size, '
                 f'{len(book_results)} books')
        plt.xlabel('Ending Token Position (Log Scale)')
        plt.ylabel('Perplexity')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.ylim(0, 55)
        
        # Legend
        if len(book_results) <= 20:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        else:
            plt.legend(['Median across all books'], loc='upper right')
        
        plt.tight_layout()
        
        # Save
        filename = f'perplexity_progression_{exp_name}.png'
        filepath = os.path.join(exp_dir, exp_name, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved perplexity plot: {filepath}")
        
        return fig  # Return figure for W&B logging
    
    @staticmethod
    def _plot_ttt_norms(book_results: List[Dict[str, Any]], exp_name: str, 
                       exp_dir: str, ppl_seq_size: int, compute_chunk_size: int, 
                       tokens_per_book: int) -> Optional[plt.Figure]:
        """Plot TTT norm progression."""
        fig = plt.figure(figsize=(14, 8))
        
        norm_position_data = {}
        valid_norms_found = False
        
        # Plot individual books
        for result in book_results:
            positions = result['seq_ending_positions']
            norms = result['seq_ttt_norms']
            
            # Filter valid norms
            valid_positions = []
            valid_norms = []
            for pos, norm in zip(positions, norms):
                if norm is not None:
                    valid_positions.append(pos)
                    valid_norms.append(norm)
                    valid_norms_found = True
                    
                    if pos not in norm_position_data:
                        norm_position_data[pos] = []
                    norm_position_data[pos].append(norm)
            
            if valid_norms:
                plt.plot(valid_positions, valid_norms, linewidth=1.0, alpha=0.1,
                        marker='+', markersize=2, label=f'Book {result["book_id"]}')
        
        if not valid_norms_found:
            plt.text(0.5, 0.5, 'No valid TTT norms found\nCheck TTT cache implementation', 
                    transform=plt.gca().transAxes, ha='center', va='center',
                    fontsize=16, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            print("[VIZ] WARNING: No valid TTT norms found for plotting")
        else:
            # Plot median norm line
            sorted_norm_positions = sorted(norm_position_data.keys())
            median_norms = [np.median(norm_position_data[pos]) for pos in sorted_norm_positions]
            
            plt.plot(sorted_norm_positions, median_norms, linewidth=2, alpha=1.0,
                    color='red', marker='o', markersize=4,
                    label='Median TTT norm', zorder=10)
            
            # Reference line
            if median_norms:
                plt.axhline(y=median_norms[0], color='blue', linestyle='--',
                           linewidth=1, alpha=0.7,
                           label=f'First norm ({median_norms[0]:.6f})')
        
        # Format plot
        plt.title(f'TTT Weight Norm Progression ({exp_name})\n'
                 f'{ppl_seq_size} tokens/seq, {compute_chunk_size} chunk size, '
                 f'{len(book_results)} books')
        plt.xlabel('Ending Token Position (Log Scale)')
        plt.ylabel('TTT Weight Norm')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Legend
        if len(book_results) <= 20 and valid_norms_found:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        elif valid_norms_found:
            plt.legend(['Median TTT norm'], loc='upper right')
        
        plt.tight_layout()
        
        # Save
        filename = f'ttt_norm_progression_{exp_name}.png'
        filepath = os.path.join(exp_dir, exp_name, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved TTT norm plot: {filepath}")
        
        return fig  # Return figure for W&B logging

    @staticmethod
    def _print_statistics(book_results: List[Dict[str, Any]]) -> None:
        """Print summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        total_sequences = sum(len(r['seq_ending_positions']) for r in book_results)
        total_tokens = sum(r['total_tokens'] for r in book_results)
        total_time = sum(r['calc_time'] for r in book_results)
        
        print(f"Books processed:     {len(book_results)}")
        print(f"Total sequences:     {total_sequences}")
        print(f"Total tokens:        {total_tokens}")
        print(f"Total calc time:     {total_time:.2f}s")
        print(f"Avg time/sequence:   {total_time/max(1, total_sequences):.4f}s")
        
        # Per-book statistics
        all_perplexities = []
        all_norms = []
        
        for result in book_results:
            book_id = result['book_id']
            perplexities = result['seq_perplexities']
            norms = [n for n in result['seq_ttt_norms'] if n is not None]
            
            all_perplexities.extend(perplexities)
            all_norms.extend(norms)
            
            if perplexities:
                ppl_stats = f"PPL: {np.median(perplexities):.3f} ± {np.std(perplexities):.3f}"
            else:
                ppl_stats = "PPL: No data"
            
            if norms:
                norm_stats = f"TTT: {np.median(norms):.6f} ± {np.std(norms):.6f}"
            else:
                norm_stats = "TTT: No data"
            
            print(f"  {book_id}: {ppl_stats}, {norm_stats}")
        
        # Overall statistics
        if all_perplexities:
            print(f"\nOverall PPL: {np.median(all_perplexities):.3f} ± {np.std(all_perplexities):.3f}")
        if all_norms:
            print(f"Overall TTT: {np.median(all_norms):.6f} ± {np.std(all_norms):.6f}")
        else:
            print("Overall TTT: No valid norms calculated")
    
    @staticmethod
    def save_results_to_file(book_results: List[Dict[str, Any]], exp_name: str, 
                            exp_dir: str, ppl_seq_size: int, compute_chunk_size: int, 
                            tokens_per_book: int, total_time: float) -> str:
        """Save detailed results to a structured text file."""
        print("[SAVE] Saving results to text file...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(exp_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ttt_perplexity_results_{exp_name}_{timestamp}.txt'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header with configuration
            f.write("# TTT Language Model Perplexity Evaluation Results\n")
            f.write(f"# Experiment: {exp_name}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Configuration: ppl_seq_size={ppl_seq_size}, chunk_size={compute_chunk_size}, tokens_per_book={tokens_per_book}\n")
            f.write(f"# Total_time: {total_time:.2f}s\n")
            f.write(f"# Books_processed: {len(book_results)}\n")
            f.write("#\n")
            
            # Overall summary statistics
            total_sequences = sum(len(r['seq_ending_positions']) for r in book_results)
            total_tokens = sum(r['total_tokens'] for r in book_results)
            all_perplexities = []
            all_norms = []
            
            for result in book_results:
                all_perplexities.extend(result['seq_perplexities'])
                all_norms.extend([n for n in result['seq_ttt_norms'] if n is not None])
            
            f.write("# SUMMARY_STATS\n")
            f.write(f"total_sequences: {total_sequences}\n")
            f.write(f"total_tokens: {total_tokens}\n")
            f.write(f"avg_time_per_sequence: {total_time/max(1, total_sequences):.6f}\n")
            
            if all_perplexities:
                f.write(f"perplexity_median: {np.median(all_perplexities):.6f}\n")
                f.write(f"perplexity_mean: {np.mean(all_perplexities):.6f}\n")
                f.write(f"perplexity_std: {np.std(all_perplexities):.6f}\n")
                f.write(f"perplexity_min: {np.min(all_perplexities):.6f}\n")
                f.write(f"perplexity_max: {np.max(all_perplexities):.6f}\n")
            else:
                f.write("perplexity_median: NaN\n")
                f.write("perplexity_mean: NaN\n")
                f.write("perplexity_std: NaN\n")
                f.write("perplexity_min: NaN\n")
                f.write("perplexity_max: NaN\n")
            
            if all_norms:
                f.write(f"ttt_norm_median: {np.median(all_norms):.8f}\n")
                f.write(f"ttt_norm_mean: {np.mean(all_norms):.8f}\n")
                f.write(f"ttt_norm_std: {np.std(all_norms):.8f}\n")
                f.write(f"ttt_norm_min: {np.min(all_norms):.8f}\n")
                f.write(f"ttt_norm_max: {np.max(all_norms):.8f}\n")
            else:
                f.write("ttt_norm_median: NaN\n")
                f.write("ttt_norm_mean: NaN\n")
                f.write("ttt_norm_std: NaN\n")
                f.write("ttt_norm_min: NaN\n")
                f.write("ttt_norm_max: NaN\n")
            
            f.write("#\n")
            
            # Per-book summary
            f.write("# BOOK_SUMMARY\n")
            f.write("# Format: book_id,sequences,tokens,calc_time,ppl_median,ppl_std,norm_median,norm_std\n")
            
            for result in book_results:
                book_id = result['book_id']
                sequences = len(result['seq_ending_positions'])
                tokens = result['total_tokens']
                calc_time = result['calc_time']
                perplexities = result['seq_perplexities']
                norms = [n for n in result['seq_ttt_norms'] if n is not None]
                
                ppl_median = np.median(perplexities) if perplexities else float('nan')
                ppl_std = np.std(perplexities) if perplexities else float('nan')
                norm_median = np.median(norms) if norms else float('nan')
                norm_std = np.std(norms) if norms else float('nan')
                
                f.write(f"{book_id},{sequences},{tokens},{calc_time:.3f},{ppl_median:.6f},{ppl_std:.6f},{norm_median:.8f},{norm_std:.8f}\n")
            
            f.write("#\n")
            
            # Detailed raw data
            f.write("# RAW_DATA\n")
            f.write("# Format: book_id,position,perplexity,ttt_norm\n")
            
            for result in book_results:
                book_id = result['book_id']
                positions = result['seq_ending_positions']
                perplexities = result['seq_perplexities']
                norms = result['seq_ttt_norms']
                
                for pos, ppl, norm in zip(positions, perplexities, norms):
                    norm_str = f"{norm:.8f}" if norm is not None else "NaN"
                    f.write(f"{book_id},{pos},{ppl:.6f},{norm_str}\n")
        
        print(f"[SAVE] Results saved to: {filepath}")
        return filepath

def setup_model_and_mesh(flags: Any) -> Tuple[Any, Any, Any, Any, Any]:
    """Set up JAX mesh, model, and compilation."""
    print("[SETUP] Initializing JAX and model...")
    
    # JAX setup
    JaxDistributedConfig.initialize(flags.jax_distributed)
    set_random_seed(flags.seed)
    
    # Mesh setup
    devices = jax.devices()
    print(f"[SETUP] Available devices: {len(devices)}")
    
    mesh_shape = (1, len(devices), 1) if len(devices) > 0 else (1, 1, 1)
    mesh_axis_names = ('dp', 'fsdp', 'mp')
    device_mesh_arr = mesh_utils.create_device_mesh(mesh_shape, devices=devices)
    mesh = Mesh(devices=device_mesh_arr, axis_names=mesh_axis_names)
    
    print(f"[SETUP] Mesh shape: {mesh_shape}, axis names: {mesh_axis_names}")
    
    # Model config
    model_config = ModelConfig.load_config(flags.load_model_config)
    
    if flags.update_model_config:
        update_dict = eval(flags.update_model_config)
        for key, value in update_dict.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise KeyError(f"Update key {key} not in model_config")
    
    # Set sequence lengths
    model_config.seq_length = flags.compute_chunk_size
    model_config.max_sequence_length = flags.compute_chunk_size
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(flags.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    model_config.vocab_size = len(tokenizer)
    model_config.eos_token_id = getattr(tokenizer, 'eos_token_id', 0)
    model_config.pad_token_id = tokenizer.pad_token_id

    model_config.use_cache = True
    
    # Model creation
    model = CausalLM(model_config, dtype=get_float_dtype_by_name(flags.dtype))
    
    print(f"[SETUP] Model config: vocab_size={model_config.vocab_size}, "
          f"max_seq_len={model_config.max_sequence_length}")
    
    return mesh, model, model_config, tokenizer

def load_model_checkpoint(flags: Any, mesh: Any, model: Any, model_config: Any) -> Any:
    """Load model checkpoint with proper sharding."""
    print("[CHECKPOINT] Loading model weights...")
    
    # Create abstract parameters for sharding
    def abstract_init_params_fn(rng_key):
        dummy_input_ids = jnp.zeros((1, model_config.max_sequence_length), dtype=jnp.int32)
        dummy_attention_mask = jnp.ones((1, model_config.max_sequence_length), dtype=jnp.int32)
        return model.init(JaxRNG(rng_key)(model_config.rng_keys()), 
                         input_ids=dummy_input_ids, 
                         attention_mask=dummy_attention_mask)["params"]
    
    abstract_params_shape = jax.eval_shape(abstract_init_params_fn, global_next_rng())
    
    # Sharding setup
    partition_rules = model_config.get_partition_rules()
    params_sharding_spec = match_partition_rules(partition_rules, abstract_params_shape)
    
    # Load checkpoint
    checkpoint_path = os.path.join(flags.exp_dir, flags.exp_name, "streaming_train_state")
    print(f"[CHECKPOINT] Loading from: {checkpoint_path}")
    
    with mesh:
        param_shard_fns, _ = make_shard_and_gather_fns(params_sharding_spec, abstract_params_shape)
        
        # First try to load the full checkpoint structure
        try:
            loaded_params = StreamingCheckpointer.load_checkpoint(
                path=checkpoint_path,
                target=None,
                shard_fns=param_shard_fns,
                remove_dict_prefix=('params',)  # Only remove 'params' prefix
            )
        except Exception as e:
            print(f"[CHECKPOINT] First attempt failed: {e}")
            # Try alternative prefix removal
            loaded_params = StreamingCheckpointer.load_checkpoint(
                path=checkpoint_path,
                target=None,
                shard_fns=param_shard_fns,
                remove_dict_prefix=None  # Don't remove any prefix
            )
            # If loaded_params has a 'params' key, extract it
            if isinstance(loaded_params, dict) and 'params' in loaded_params:
                loaded_params = loaded_params['params']
    
    # Convert to FrozenDict
    loaded_params = freeze(loaded_params)
    print("[CHECKPOINT] Parameters loaded and frozen successfully")
    
    # Debug: print the structure to understand what we have
    if flags.debug_mode:
        print(f"[CHECKPOINT] Loaded params type: {type(loaded_params)}")
        if hasattr(loaded_params, 'keys'):
            print(f"[CHECKPOINT] Loaded params keys: {list(loaded_params.keys())}")
        print(f"[CHECKPOINT] Abstract params type: {type(abstract_params_shape)}")
        if hasattr(abstract_params_shape, 'keys'):
            print(f"[CHECKPOINT] Abstract params keys: {list(abstract_params_shape.keys())}")
    
    return loaded_params, params_sharding_spec

def compile_model_function(mesh: Any, model: Any, params_sharding_spec: Any) -> Any:
    """Compile the model forward function."""
    print("[COMPILE] Compiling model function...")
    
    def model_forward_fn(params, inputs, cache_variables, model_rngs):
        variables = {'params': params}
        
        if cache_variables is not None:
            for cache_type, cache_variables in cache_variables.items():
                variables[cache_type] = cache_variables
        
        return model.apply(
            variables,
            input_ids=inputs,
            deterministic=True,
            mutable=['ttt_cache', 'conv_cache', 'pre_conv_cache', 
            "mini_batch_counter",
            'ttt_cache_medium', 'ttt_cache_slow'
            ],
            rngs=model_rngs
        )
    
    with mesh:
        compiled_fn = pjit.pjit(
            model_forward_fn,
            in_shardings=(params_sharding_spec, PS(), PS(), PS()),
            out_shardings=(PS(), PS())
        )
    
    print("[COMPILE] Model compilation completed")
    return compiled_fn

def debug_ttt_functionality(mesh: Any, model: Any, model_config: Any, 
                          compiled_fn: Any, params: Any, tokenizer: Any) -> None:
    """Debug TTT functionality with a simple test."""
    print("\n" + "="*60)
    print("TTT FUNCTIONALITY DEBUG")
    print("="*60)
    
    # Create test input
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    test_tokens = tokenizer(test_text, return_tensors="np", return_attention_mask=False)
    test_input = jnp.array(test_tokens.input_ids)
    
    if test_input.ndim == 1:
        test_input = jnp.expand_dims(test_input, axis=0)
    
    # Pad to model length
    if test_input.shape[1] < model_config.max_sequence_length:
        padding_len = model_config.max_sequence_length - test_input.shape[1]
        padding = jnp.full((1, padding_len), model_config.pad_token_id, dtype=test_input.dtype)
        test_input = jnp.concatenate([test_input, padding], axis=1)
    
    print(f"[DEBUG] Test input shape: {test_input.shape}")
    
    norm_calc = TTTNormCalculator()
    
    with mesh:
        try:
            # First forward pass
            print("[DEBUG] First forward pass...")
            
            # Generate PRNG keys outside the compiled function
            rng = JaxRNG(global_next_rng())
            model_rngs = rng(model.config.rng_keys())
            
            outputs1, vars1 = compiled_fn(params, test_input, None, model_rngs)
            
            print(f"[DEBUG] Output type: {type(outputs1)}")
            print(f"[DEBUG] Variables keys: {list(vars1.keys())}")
            
            for key, value in vars1.items():
                print(f"[DEBUG] {key}: {type(value)}")
                if hasattr(value, 'keys'):
                    print(f"[DEBUG]   Sub-keys: {list(value.keys())}")
            
            # Test TTT norm calculation
            if 'ttt_cache' in vars1:
                print("[DEBUG] Found ttt_cache! Inspecting structure...")
                
                # Detailed FrozenDict inspection
                norm_calc.inspect_frozen_dict_structure(vars1['ttt_cache'], max_depth=4)
                
                print("[DEBUG] Testing TTT norm calculation...")
                norm1 = norm_calc.calculate_norm(vars1['ttt_cache'], debug=True)
                print(f"[DEBUG] First pass TTT norm: {norm1}")
                
                if norm1 is None:
                    print("[DEBUG] TTT norm is None - investigating why...")
                    # Let's try to manually traverse and see what's happening
                    from flax.core.frozen_dict import FrozenDict
                    cache = vars1['ttt_cache']
                    if isinstance(cache, FrozenDict):
                        print(f"[DEBUG] FrozenDict has keys: {list(cache.keys())}")
                        for key, value in cache.items():
                            print(f"[DEBUG]   {key}: {type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"[DEBUG]     Shape: {value.shape}, Size: {value.size}")
                                # Check if array has any non-zero values
                                non_zero = jnp.count_nonzero(value)
                                print(f"[DEBUG]     Non-zero elements: {non_zero}")
                
            else:
                print("[DEBUG] No ttt_cache found in first pass")
            
            # Second forward pass with cache
            print("\n[DEBUG] Second forward pass with cache...")
            cache = {}
            for cache_type in ['ttt_cache', 'pre_conv_cache', 'conv_cache',
            "ttt_cache_medium", "ttt_cache_slow"
            ]:
                if cache_type in vars1:
                    cache[cache_type] = vars1[cache_type]
            
            if cache:
                # Generate new PRNG keys for second pass
                rng2 = JaxRNG(global_next_rng())
                model_rngs2 = rng2(model.config.rng_keys())
                
                outputs2, vars2 = compiled_fn(params, test_input, cache, model_rngs2)
                if 'ttt_cache' in vars2:
                    print("[DEBUG] TTT cache from second pass:")
                    norm_calc.inspect_frozen_dict_structure(vars2['ttt_cache'], max_depth=3)
                    norm2 = norm_calc.calculate_norm(vars2['ttt_cache'], debug=True)
                    print(f"[DEBUG] Second pass TTT norm: {norm2}")
                    
                    # Compare norms if both exist
                    if norm1 is not None and norm2 is not None:
                        print(f"[DEBUG] Norm change: {norm1:.6f} -> {norm2:.6f} (delta: {norm2-norm1:.6f})")
                else:
                    print("[DEBUG] No ttt_cache found in second pass")
            else:
                print("[DEBUG] No cache to use for second pass")
                
        except Exception as e:
            print(f"[DEBUG] ERROR in debug test: {e}")
            traceback.print_exc()

def load_wandb_run_id(exp_dir: str, exp_name: str) -> Optional[str]:
    """Load W&B run ID from the training run."""
    wandb_id_file = os.path.join(exp_dir, exp_name, "wandb_run_id.txt")
    if os.path.exists(wandb_id_file):
        with open(wandb_id_file, 'r') as f:
            wandb_run_id = f.read().strip()
        print(f"[WANDB] Found existing W&B run ID: {wandb_run_id}")
        return wandb_run_id
    else:
        print(f"[WANDB] No W&B run ID file found at: {wandb_id_file}")
        return None

def initialize_wandb(flags: Any) -> Optional[Any]:
    """Initialize W&B logging by resuming existing run or creating new one."""
    if not flags.use_wandb or not flags.log_to_wandb:
        print("[WANDB] W&B logging disabled")
        return None
    
    try:
        # Try to resume existing run from training
        wandb_run_id = load_wandb_run_id(flags.exp_dir, flags.exp_name)
        
        if wandb_run_id:
            # Resume existing run with minimal settings to avoid overriding
            print("[WANDB] Resuming existing W&B run...")
            wandb_run = wandb.init(
                project=flags.wandb_project,
                id=wandb_run_id,
                resume="allow",
                # Don't set name or tags to avoid overriding
                settings=wandb.Settings(
                    resume="allow",   # Don't override config when resuming
                    save_code=False,  # Don't save code to avoid command override
                    program_relpath=None,  # Don't set program path
                    program=None,  # Don't set program name
                    _disable_stats=True,  # Disable system stats collection
                    _disable_meta=True,   # Disable metadata collection
                )
            )
            
            # Log perplexity configuration as separate metrics instead of config
            wandb_run.log({
                "perplexity_config/compute_chunk_size": flags.compute_chunk_size,
                "perplexity_config/num_books": flags.num_books,
                "perplexity_config/tokens_per_book": flags.tokens_per_book,
                "perplexity_config/ppl_seq_size": flags.ppl_seq_size,
                "perplexity_config/debug_mode": flags.debug_mode,
                "perplexity_config/evaluation_started": True,
            })
            
            print(f"[WANDB] Successfully resumed W&B run: {wandb_run_id}")
        else:
            # Create new run for perplexity evaluation
            print("[WANDB] Creating new W&B run for perplexity evaluation...")
            wandb_run = wandb.init(
                project=flags.wandb_project,
                name=f"{flags.exp_name}_perplexity",
                tags=["perplexity_evaluation"],
                config={
                    "exp_name": flags.exp_name,
                    "compute_chunk_size": flags.compute_chunk_size,
                    "books_dir": flags.books_dir,
                    "num_books": flags.num_books,
                    "tokens_per_book": flags.tokens_per_book,
                    "ppl_seq_size": flags.ppl_seq_size,
                    "debug_mode": flags.debug_mode,
                }
            )
            print(f"[WANDB] Created new W&B run: {wandb_run.id}")
        
        return wandb_run
        
    except Exception as e:
        print(f"[WANDB] Failed to initialize W&B: {e}")
        return None

def log_perplexity_results_to_wandb(wandb_run: Any, book_results: List[Dict[str, Any]], 
                                   flags: Any) -> None:
    """Log perplexity evaluation results to W&B."""
    if wandb_run is None:
        return
    
    try:
        print("[WANDB] Logging perplexity results...")
        
        # Calculate overall statistics
        all_perplexities = []
        all_norms = []
        total_sequences = 0
        total_tokens = 0
        total_time = 0
        
        for result in book_results:
            all_perplexities.extend(result['seq_perplexities'])
            all_norms.extend([n for n in result['seq_ttt_norms'] if n is not None])
            total_sequences += len(result['seq_ending_positions'])
            total_tokens += result['total_tokens']
            total_time += result['calc_time']
        
        # Overall metrics
        log_dict = {
            "perplexity_evaluation/books_processed": len(book_results),
            "perplexity_evaluation/total_sequences": total_sequences,
            "perplexity_evaluation/total_tokens": total_tokens,
            "perplexity_evaluation/total_time_seconds": total_time,
            "perplexity_evaluation/avg_time_per_sequence": total_time / max(1, total_sequences),
        }
        
        # Perplexity statistics
        if all_perplexities:
            log_dict.update({
                "perplexity_evaluation/perplexity_median": np.median(all_perplexities),
                "perplexity_evaluation/perplexity_mean": np.mean(all_perplexities),
                "perplexity_evaluation/perplexity_std": np.std(all_perplexities),
                "perplexity_evaluation/perplexity_min": np.min(all_perplexities),
                "perplexity_evaluation/perplexity_max": np.max(all_perplexities),
            })
        
        # TTT norm statistics
        if all_norms:
            log_dict.update({
                "perplexity_evaluation/ttt_norm_median": np.median(all_norms),
                "perplexity_evaluation/ttt_norm_mean": np.mean(all_norms),
                "perplexity_evaluation/ttt_norm_std": np.std(all_norms),
                "perplexity_evaluation/ttt_norm_min": np.min(all_norms),
                "perplexity_evaluation/ttt_norm_max": np.max(all_norms),
                "perplexity_evaluation/valid_ttt_norms": len(all_norms),
            })
        else:
            log_dict["perplexity_evaluation/valid_ttt_norms"] = 0
        
        # Configuration info
        log_dict.update({
            "perplexity_evaluation/chunk_size": flags.compute_chunk_size,
            "perplexity_evaluation/ppl_seq_size": flags.ppl_seq_size,
            "perplexity_evaluation/tokens_per_book": flags.tokens_per_book,
        })
        
        # Log all metrics
        wandb_run.log(log_dict)
        
        # Log per-book metrics as a table
        book_data = []
        for result in book_results:
            perplexities = result['seq_perplexities']
            norms = [n for n in result['seq_ttt_norms'] if n is not None]
            
            book_row = {
                "book_id": result['book_id'],
                "sequences": len(result['seq_ending_positions']),
                "tokens": result['total_tokens'],
                "calc_time": result['calc_time'],
                "ppl_median": np.median(perplexities) if perplexities else None,
                "ppl_std": np.std(perplexities) if perplexities else None,
                "norm_median": np.median(norms) if norms else None,
                "norm_std": np.std(norms) if norms else None,
                "valid_norms": len(norms),
            }
            book_data.append(book_row)
        
        # Create W&B table
        table = wandb.Table(
            columns=["book_id", "sequences", "tokens", "calc_time", 
                    "ppl_median", "ppl_std", "norm_median", "norm_std", "valid_norms"],
            data=[[row[col] for col in ["book_id", "sequences", "tokens", "calc_time", 
                                      "ppl_median", "ppl_std", "norm_median", "norm_std", "valid_norms"]] 
                  for row in book_data]
        )
        wandb_run.log({"perplexity_evaluation/book_results": table})
        
        print(f"[WANDB] Successfully logged perplexity results to W&B")
        
    except Exception as e:
        print(f"[WANDB] Error logging to W&B: {e}")

def main(argv):
    """Main evaluation function."""
    start_time = time.time()
    
    print(f"[MAIN] Starting TTT perplexity evaluation")
    print(f"[MAIN] JAX backend: {jax.default_backend()}")
    
    # Get flags
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    
    # Initialize W&B
    wandb_run = initialize_wandb(FLAGS)
    
    # Setup
    mesh, model, model_config, tokenizer = setup_model_and_mesh(FLAGS)
    loaded_params, params_sharding_spec = load_model_checkpoint(FLAGS, mesh, model, model_config)
    compiled_fn = compile_model_function(mesh, model, params_sharding_spec)
    
    # Debug TTT functionality if requested
    if FLAGS.debug_mode:
        debug_ttt_functionality(mesh, model, model_config, compiled_fn, loaded_params, tokenizer)
    
    # Load books
    books = BookLoader.load_books(
        FLAGS.books_dir, tokenizer, FLAGS.num_books, 
        FLAGS.tokens_per_book, FLAGS.skip_start_chars
    )
    
    if not books:
        print("[MAIN] ERROR: No valid books loaded. Exiting.")
        if wandb_run:
            wandb_run.finish()
        return
    
    # Initialize calculator
    norm_calculator = TTTNormCalculator()
    perplexity_calculator = PerplexityCalculator(
        model, model_config, compiled_fn, norm_calculator, debug=FLAGS.debug_mode
    )
    
    # Process books
    print(f"\n[MAIN] Processing {len(books)} books...")
    book_results = []
    
    with mesh:
        for book_idx, (book_id, book_text) in enumerate(books):
            print(f"\n[MAIN] Book {book_idx+1}/{len(books)}: {book_id}")
            
            # Tokenize
            tokenized = tokenizer(book_text, return_tensors="np", return_attention_mask=False)
            input_ids = jnp.array(tokenized.input_ids)
            
            # Calculate perplexity and norms
            result = perplexity_calculator.calculate_book_perplexity(
                loaded_params, input_ids, book_id, FLAGS.tokens_per_book,
                FLAGS.ppl_seq_size, FLAGS.compute_chunk_size
            )
            
            book_results.append(result)
            print(f"[MAIN] Book {book_id}: {len(result['seq_ending_positions'])} sequences completed")
    
    # Visualize results
    if book_results:
        # Log to W&B before plotting
        if wandb_run:
            log_perplexity_results_to_wandb(wandb_run, book_results, FLAGS)
        
        ResultsVisualizer.plot_results(
            book_results, FLAGS.exp_name, FLAGS.exp_dir,
            FLAGS.ppl_seq_size, FLAGS.compute_chunk_size, FLAGS.tokens_per_book,
            wandb_run
        )
        
        # Save results to text file
        total_time = time.time() - start_time
        results_file = ResultsVisualizer.save_results_to_file(
            book_results, FLAGS.exp_name, FLAGS.exp_dir,
            FLAGS.ppl_seq_size, FLAGS.compute_chunk_size, FLAGS.tokens_per_book,
            total_time
        )
        print(f"[MAIN] Results saved to: {results_file}")
    else:
        print("[MAIN] No results to visualize or save")
    
    total_time = time.time() - start_time
    print(f"\n[MAIN] Evaluation completed in {total_time:.2f} seconds")
    
    # Finish W&B run
    if wandb_run:
        wandb_run.finish()
        print("[WANDB] W&B run finished")

if __name__ == "__main__":
    mlxu.run(main)