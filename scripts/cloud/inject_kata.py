#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Inject one or many katas into the pod's corpus during a live run.

Workflow per kata:
  1. Validate locally — must have metadata.json + setup.lisp + tests.lisp
     and a non-empty `instruction` field in metadata.
  2. SCP the directory to /workspace/katas/<bucket>/<kata_id>/ on the pod
     (default bucket: cl-ds; pass --bucket creative to use that mix).
  3. Symlink the kata into /workspace/macro-gym/katas/<kata_id> so
     `MacroEnv(kata_id=...)` can resolve it.
  4. Verify by spawning a tiny python on the pod that does
     `MacroEnv(kata_id=...).reset()` and reports success.

What this does NOT do:
  • It doesn't touch the running trainer's frozen `train_dataset` — new
    katas only enter the training mix at the next bootstrap. Use this
    helper to pre-stage curated katas during a long run; restart the
    trainer between runs to absorb them.
  • It doesn't update the kata count guard in run_grpo.sh — that's a
    pure read from /workspace/katas/.

Usage:
  scripts/cloud/inject_kata.py POD_ID KATA_DIR [KATA_DIR ...]
  scripts/cloud/inject_kata.py POD_ID path/to/dir_of_katas --batch
  scripts/cloud/inject_kata.py POD_ID kata1 kata2 --bucket creative

Examples:
  # one kata
  scripts/cloud/inject_kata.py qecfy5ronmbvih ~/katas/cl-ds-99001

  # many katas in a parent dir
  scripts/cloud/inject_kata.py qecfy5ronmbvih ~/katas-batch --batch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ─── shell helpers ───────────────────────────────────────────────────

def _fail(msg: str) -> None:
    print(f"  xx {msg}", file=sys.stderr)
    sys.exit(1)


def _pod_ssh_info(pod_id: str) -> tuple[str, int]:
    try:
        out = subprocess.check_output(
            ["runpodctl", "ssh", "info", pod_id], text=True
        )
        info = json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError,
            FileNotFoundError) as e:
        _fail(f"couldn't resolve SSH info for pod {pod_id}: {e}")
    ip   = info.get("ip")
    port = info.get("port")
    if not ip or not port:
        _fail(f"pod info missing ip/port: {info}")
    return str(ip), int(port)


# ─── kata validation ─────────────────────────────────────────────────

REQUIRED_FILES = ("metadata.json", "setup.lisp", "tests.lisp")


def _validate(kata_dir: Path) -> str:
    """Return kata_id (= leaf dir name) or raise via _fail."""
    if not kata_dir.is_dir():
        _fail(f"{kata_dir} is not a directory")
    for f in REQUIRED_FILES:
        if not (kata_dir / f).is_file():
            _fail(f"{kata_dir.name}: missing {f}")
    try:
        meta = json.loads((kata_dir / "metadata.json").read_text())
    except json.JSONDecodeError as e:
        _fail(f"{kata_dir.name}: metadata.json not valid JSON: {e}")
    if not (meta.get("instruction") or "").strip():
        _fail(f"{kata_dir.name}: metadata.json 'instruction' is blank")
    return kata_dir.name


# ─── per-pod injection ───────────────────────────────────────────────

def _inject(kata_dir: Path, kata_id: str, *,
            ssh_key: Path, ip: str, port: int, bucket: str) -> bool:
    """SCP + symlink + verify one kata. Returns True on success."""
    base_ssh = [
        "ssh", "-i", str(ssh_key),
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(port), f"root@{ip}",
    ]
    base_scp = [
        "scp", "-i", str(ssh_key),
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-r", "-P", str(port),
    ]

    target_root = f"/workspace/katas/{bucket}"
    target_kata = f"{target_root}/{kata_id}"
    sym         = f"/workspace/macro-gym/katas/{kata_id}"

    # Ensure target bucket exists on pod
    subprocess.run(base_ssh + [f"mkdir -p '{target_root}'"], check=True)

    # Warn if overwriting
    check = subprocess.run(
        base_ssh + [f"test -d '{target_kata}' && echo EXISTS || echo NEW"],
        capture_output=True, text=True,
    )
    overwriting = "EXISTS" in check.stdout
    if overwriting:
        # Wipe so SCP -r doesn't nest into target_kata/kata_id.
        subprocess.run(base_ssh + [f"rm -rf '{target_kata}'"], check=True)
        print(f"  ⚠ overwriting existing kata at {target_kata}")

    # SCP
    print(f"  scp {kata_dir.name} -> {target_kata}")
    r = subprocess.run(base_scp + [str(kata_dir), f"root@{ip}:{target_kata}"])
    if r.returncode != 0:
        print(f"  xx scp failed (exit {r.returncode})", file=sys.stderr)
        return False

    # Symlink into macro-gym katas
    r = subprocess.run(base_ssh + [f"ln -sfn '{target_kata}' '{sym}'"])
    if r.returncode != 0:
        print(f"  xx symlink failed (exit {r.returncode})", file=sys.stderr)
        return False

    # Verify with macro-gym
    verify_cmd = (
        f"python -c \"import sys; "
        f"from macro_gym import MacroEnv; "
        f"e = MacroEnv(kata_id='{kata_id}'); "
        f"obs, info = e.reset(); "
        f"sys.exit(0 if info.get('kata_id') == '{kata_id}' else 1)\""
    )
    v = subprocess.run(base_ssh + [verify_cmd],
                       capture_output=True, text=True)
    if v.returncode != 0:
        print(f"  xx macro-gym verification failed:\n{v.stderr.strip()}",
              file=sys.stderr)
        return False

    print(f"  ✓ {kata_id} injected")
    return True


# ─── main ────────────────────────────────────────────────────────────

def _expand_paths(paths: list[Path], batch: bool) -> list[Path]:
    """In batch mode, each PATH is a parent dir; collect kata subdirs."""
    out: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if not p.is_dir():
            _fail(f"{p} is not a directory")
        if batch:
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / "metadata.json").is_file():
                    out.append(sub)
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pod_id", help="RunPod pod id (e.g. qecfy5ronmbvih)")
    ap.add_argument("kata_dirs", nargs="+", type=Path,
                    help="kata directory or directories")
    ap.add_argument("--batch", action="store_true",
                    help="each kata_dir is actually a parent of kata dirs")
    ap.add_argument("--bucket", default="cl-ds",
                    help="kata bucket under /workspace/katas/ (default: cl-ds)")
    ap.add_argument("--ssh-key", type=Path,
                    default=Path("~/.runpod/ssh/RunPod-Key-Go").expanduser(),
                    help="SSH private key path")
    args = ap.parse_args()

    ssh_key = args.ssh_key.expanduser()
    if not ssh_key.is_file():
        _fail(f"SSH key not found: {ssh_key}")

    katas = _expand_paths(args.kata_dirs, batch=args.batch)
    if not katas:
        _fail("no katas to inject")
    print(f"  found {len(katas)} kata(s) to inject")

    # Validate ALL before touching the pod — fail fast.
    kata_ids = [_validate(k) for k in katas]

    # Resolve pod SSH
    ip, port = _pod_ssh_info(args.pod_id)
    print(f"  pod {args.pod_id}: {ip}:{port}")

    # Inject one at a time. Stop on first failure so it's easy to debug.
    ok = 0
    for k, kid in zip(katas, kata_ids):
        if _inject(k, kid, ssh_key=ssh_key, ip=ip, port=port, bucket=args.bucket):
            ok += 1
        else:
            print(f"\n  xx aborting after {ok}/{len(katas)} successful "
                  f"injections", file=sys.stderr)
            return 1

    print(f"\n  ✓ all {ok} katas injected. They will enter the training mix on "
          f"the NEXT bootstrap; the current trainer keeps its frozen dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
