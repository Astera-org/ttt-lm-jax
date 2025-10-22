#!/usr/bin/env python3
"""Utility for uploading experiment checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError


def _build_operations(exp_dir: Path, path_in_repo: str | None) -> list[CommitOperationAdd]:
    """Create hub commit operations for every file under the experiment directory."""
    operations: list[CommitOperationAdd] = []
    base_repo_path = Path(path_in_repo) if path_in_repo else None

    for item in sorted(exp_dir.rglob("*")):
        if not item.is_file():
            continue

        repo_path = item.relative_to(exp_dir)
        if base_repo_path:
            repo_path = base_repo_path / repo_path

        operations.append(
            CommitOperationAdd(
                path_in_repo=str(repo_path).replace(os.sep, "/"),
                path_or_fileobj=str(item),
            )
        )

    return operations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an experiment directory (all checkpoints) to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Path to the local experiment directory, e.g. current_exp/ttt_layer_...",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hub repository in the form 'org/name' or 'user/name'.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hub repository type. Defaults to 'model'.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional subdirectory inside the Hub repo to store the upload.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message. Defaults to 'Add checkpoints from <exp name>'.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch or revision to push to. Uses the repo's default branch when omitted.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private when it does not exist yet.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not attempt to create the repository if it is missing.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment directory '{exp_dir}' does not exist or is not a directory.")

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Missing Hugging Face token. Pass --token or set HF_TOKEN in the environment.")

    api = HfApi(token=token)

    if not args.no_create:
        try:
            create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                private=args.private,
                exist_ok=True,
                token=token,
            )
        except HfHubHTTPError as err:
            raise SystemExit(f"Failed to ensure repository '{args.repo_id}': {err}") from err

    operations = _build_operations(exp_dir, args.path_in_repo or None)
    if not operations:
        raise SystemExit(f"No files found under '{exp_dir}'. Nothing to upload.")

    commit_message = args.commit_message or f"Add checkpoints from {exp_dir.name}"

    try:
        commit_info = api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            operations=operations,
            commit_message=commit_message,
            revision=args.revision,
        )
    except HfHubHTTPError as err:
        raise SystemExit(f"Upload failed: {err}") from err

    commit_ref = (
        getattr(commit_info, "commit_id", None)
        or getattr(commit_info, "oid", None)
        or getattr(commit_info, "commit_url", None)
    )

    if isinstance(commit_ref, str) and commit_ref.startswith("http"):
        commit_ref = commit_ref.rstrip("/").split("/")[-1]

    print(
        f"Uploaded {len(operations)} files from '{exp_dir}' to {args.repo_id}"
        f" (commit {commit_ref or 'unknown'})."
    )


if __name__ == "__main__":
    main()
