#!/usr/bin/env python3
"""Batch perplexity evaluation for every saved checkpoint in an experiment.

This utility scans the experiment directory for step subfolders, ensures each
contains a ``streaming_train_state`` entry, and then drives ``test_perplexity.py``
with the requested evaluation settings. Existing result files (matching the
requested ``tokens_per_book``) are left untouched so the script is restartable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class EvalSpec:
    exp_root: Path
    load_model_config: str
    update_model_config: str
    tokens_per_book: int
    ppl_seq_size: int
    compute_chunk_size: int
    books_dir: Path
    num_books: int
    tokenizer_name: str
    dtype: str
    skip_start_chars: int
    use_wandb: bool
    python_executable: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        type=Path,
        required=True,
        help="Path to the experiment root (e.g. current_exp/ttt_layer_nobias_frobenius-linear-350m-books-2k)",
    )
    parser.add_argument(
        "--load-model-config",
        required=True,
        help="Model config identifier to pass to test_perplexity.py (e.g. 350m-TTT)",
    )
    parser.add_argument(
        "--update-model-config",
        required=True,
        help="Update dict string for ModelConfig (must be valid Python expression)",
    )
    parser.add_argument(
        "--tokens-per-book",
        type=int,
        default=1_024_000,
        help="Number of tokens per book to evaluate (defaults to 500x 2048 windows)",
    )
    parser.add_argument(
        "--ppl-seq-size",
        type=int,
        default=2048,
        help="Sequence length used for perplexity aggregation",
    )
    parser.add_argument(
        "--compute-chunk-size",
        type=int,
        default=2048,
        help="Chunk size used during forward passes",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=Path("data/gutenberg/top_long_books"),
        help="Directory containing the evaluation book .txt files",
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=20,
        help="How many books to evaluate per checkpoint",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer identifier",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["fp32", "bfloat16"],
        help="Compute dtype to pass through",
    )
    parser.add_argument(
        "--skip-start-chars",
        type=int,
        default=1024,
        help="Characters to trim from the beginning of each book",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the latest checkpoint stored directly under exp_root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a matching results file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands that would be executed",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for launching test_perplexity.py",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B logging during the sweep (default: disabled)",
    )
    return parser.parse_args()


def iter_checkpoint_dirs(exp_root: Path, include_base: bool) -> Iterable[Path]:
    if include_base:
        base_ckpt = exp_root / "streaming_train_state"
        if base_ckpt.exists():
            yield exp_root
    for step_dir in sorted(exp_root.glob("step_*"), key=_step_sort_key):
        if any(step_dir.glob("streaming_train_state*")):
            yield step_dir


def _step_sort_key(path: Path) -> int:
    try:
        suffix = path.name.split("_")[1]
        return int(suffix)
    except (IndexError, ValueError):
        return sys.maxsize


def ensure_symlink(step_dir: Path) -> None:
    candidates = sorted(step_dir.glob("streaming_train_state*"))
    if not candidates:
        raise FileNotFoundError(f"No streaming_train_state* file in {step_dir}")

    # Prefer precise match with numeric suffix, falling back to plain file.
    target = None
    for cand in candidates:
        if cand.name == "streaming_train_state":
            target = cand
            break
        if cand.name.startswith("streaming_train_state_"):
            target = cand
            break

    if target is None:
        raise FileNotFoundError(f"Unable to determine checkpoint file inside {step_dir}")

    link_path = step_dir / "streaming_train_state"
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return
        link_path.unlink()

    if target.name == "streaming_train_state":
        return  # Already in the desired shape.

    link_path.symlink_to(target.name)


def has_results(step_dir: Path, tokens_per_book: int, compute_chunk_size: int, ppl_seq_size: int) -> bool:
    pattern = f"ttt_perplexity_results_*_{ppl_seq_size}seqsize_{compute_chunk_size}compsize_{tokens_per_book}tokens_*.txt"
    return any(step_dir.glob(pattern))


def build_command(spec: EvalSpec, checkpoint_dir: Path) -> List[str]:
    exp_dir = spec.exp_root
    exp_name = checkpoint_dir.name

    cmd = [
        spec.python_executable,
        "test_perplexity.py",
        "--exp_dir",
        str(exp_dir),
        "--exp_name",
        exp_name,
        "--books_dir",
        str(spec.books_dir),
        "--num_books",
        str(spec.num_books),
        "--tokens_per_book",
        str(spec.tokens_per_book),
        "--compute_chunk_size",
        str(spec.compute_chunk_size),
        "--ppl_seq_size",
        str(spec.ppl_seq_size),
        "--skip_start_chars",
        str(spec.skip_start_chars),
        "--load_model_config",
        spec.load_model_config,
        "--update_model_config",
        spec.update_model_config,
        "--tokenizer_name",
        spec.tokenizer_name,
        "--dtype",
        spec.dtype,
        f"--use_wandb={'True' if spec.use_wandb else 'False'}",
        f"--log_to_wandb={'True' if spec.use_wandb else 'False'}",
    ]
    return cmd


def main() -> None:
    args = parse_args()
    spec = EvalSpec(
        exp_root=args.exp_root.resolve(),
        load_model_config=args.load_model_config,
        update_model_config=args.update_model_config,
        tokens_per_book=args.tokens_per_book,
        ppl_seq_size=args.ppl_seq_size,
        compute_chunk_size=args.compute_chunk_size,
        books_dir=args.books_dir.resolve(),
        num_books=args.num_books,
        tokenizer_name=args.tokenizer_name,
        dtype=args.dtype,
        skip_start_chars=args.skip_start_chars,
        use_wandb=args.use_wandb,
        python_executable=args.python,
    )

    if not spec.books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {spec.books_dir}")

    checkpoints = list(iter_checkpoint_dirs(spec.exp_root, args.include_base))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints discovered under {spec.exp_root}")

    print(f"Discovered {len(checkpoints)} checkpoint directories under {spec.exp_root}")

    for ckpt_dir in checkpoints:
        ensure_symlink(ckpt_dir)
        already_done = has_results(
            ckpt_dir,
            spec.tokens_per_book,
            spec.compute_chunk_size,
            spec.ppl_seq_size,
        )
        if already_done and not args.force:
            print(f"[SKIP] Results already present for {ckpt_dir.name}")
            continue

        cmd = build_command(spec, ckpt_dir)
        print(f"[RUN] {' '.join(cmd)}")

        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Command failed for {ckpt_dir.name} with code {exc.returncode}")
            break


if __name__ == "__main__":
    main()
