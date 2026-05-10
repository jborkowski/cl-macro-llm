#!/usr/bin/env python3
"""Push the trained LoRA adapter to a HuggingFace repo.

Reads the adapter from ./output/final_adapter/ (or $OUTPUT_DIR/final_adapter)
and pushes it to the repo named in $HF_REPO using $HF_TOKEN.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


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

    print(f"Uploading {adapter_dir} → https://huggingface.co/{repo}")
    create_repo(repo, token=token, exist_ok=True, repo_type="model", private=False)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo,
        repo_type="model",
        commit_message="Upload LoRA adapter from cl-macro-llm training run",
    )

    print(f"\nAdapter pushed to: https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
