#!/usr/bin/env python3
"""Push the trained LoRA adapter to a HuggingFace repo.

Reads the adapter from ./output/final_adapter/ (or $OUTPUT_DIR/final_adapter)
and pushes it to the repo named in $HF_REPO using $HF_TOKEN. After upload,
re-queries the repo via the HF API and verifies the expected adapter files
are present, so the run summary doesn't claim success on a partial push.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REQUIRED_FILES = ("adapter_config.json",)
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def main() -> int:
    repo = os.environ.get("HF_REPO")
    token = os.environ.get("HF_TOKEN")
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    adapter_dir = Path(output_dir) / "final_adapter"

    if not repo:
        print("HF_REPO is not set — skipping upload.", file=sys.stderr)
        return 0
    if not token:
        print("HF_TOKEN is not set — cannot upload.", file=sys.stderr)
        return 1
    if not adapter_dir.exists():
        print(f"Adapter directory not found: {adapter_dir}", file=sys.stderr)
        return 1

    local_files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    print(f"Uploading {adapter_dir} ({len(local_files)} files) → https://huggingface.co/{repo}")
    for name in local_files:
        size = (adapter_dir / name).stat().st_size
        print(f"  - {name} ({size / 1024 / 1024:.1f} MB)")

    create_repo(repo, token=token, exist_ok=True, repo_type="model", private=False)

    api = HfApi(token=token)
    commit_info = api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo,
        repo_type="model",
        commit_message="Upload LoRA adapter from cl-macro-llm training run",
    )

    remote_files = set(api.list_repo_files(repo_id=repo, repo_type="model"))

    missing = [f for f in REQUIRED_FILES if f not in remote_files]
    if not any(w in remote_files for w in ADAPTER_WEIGHTS):
        missing.append(f"one of {ADAPTER_WEIGHTS!r}")

    print("\nRemote repo contents:")
    for name in sorted(remote_files):
        marker = " (local)" if name in local_files else ""
        print(f"  - {name}{marker}")

    if missing:
        print(
            f"\nUPLOAD VERIFY FAILED — missing on remote: {missing}",
            file=sys.stderr,
        )
        return 1

    commit_url = getattr(commit_info, "commit_url", None) or getattr(
        commit_info, "pr_url", None
    )
    print("\nUpload verified.")
    print(f"  repo:   https://huggingface.co/{repo}")
    if commit_url:
        print(f"  commit: {commit_url}")
    print(f"  files:  {len(remote_files)} on remote, {len(local_files)} pushed from {adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
