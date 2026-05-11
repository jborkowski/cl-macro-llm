#!/bin/bash
# Install a pinned SBCL on the pod from the official Linux binary tarball.
#
# Why not `apt-get install sbcl`? On Ubuntu 22.04 you get SBCL 2.1.x,
# which is old enough that some library macros (and our own kata
# expansions) miss recent reader / pretty-printer fixes. macro-gym
# expects a modern SBCL.
#
# Skips if a sufficiently recent `sbcl` is already on PATH.
#
# Usage:
#     bash scripts/cloud/install_sbcl.sh
#
# Env overrides:
#     SBCL_VERSION   default 2.4.0
#     SBCL_ARCH      default x86-64
#     INSTALL_ROOT   default /usr/local
#     SBCL_TMPDIR    default /tmp/sbcl-install

set -euo pipefail

SBCL_VERSION="${SBCL_VERSION:-2.4.0}"
INSTALL_ROOT="${INSTALL_ROOT:-/usr/local}"
SBCL_TMPDIR="${SBCL_TMPDIR:-/tmp/sbcl-install}"

# Auto-detect arch: SourceForge ships sbcl-${ver}-x86-64-linux-binary.tar.bz2
# and sbcl-${ver}-arm64-linux-binary.tar.bz2. Anything else, the user has
# to pass SBCL_ARCH explicitly.
if [[ -z "${SBCL_ARCH:-}" ]]; then
    case "$(uname -m)" in
        x86_64|amd64) SBCL_ARCH="x86-64" ;;
        aarch64|arm64) SBCL_ARCH="arm64" ;;
        *) echo "ERROR: unknown machine arch $(uname -m); set SBCL_ARCH manually."; exit 1 ;;
    esac
fi

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ─── Already installed? ───────────────────────────────────────────────
if command -v sbcl >/dev/null 2>&1; then
    HAVE_VERSION="$(sbcl --version | awk '{print $2}')"
    MIN_MAJOR="$(echo "$SBCL_VERSION" | cut -d. -f1)"
    MIN_MINOR="$(echo "$SBCL_VERSION" | cut -d. -f2)"
    HAVE_MAJOR="$(echo "$HAVE_VERSION" | cut -d. -f1)"
    HAVE_MINOR="$(echo "$HAVE_VERSION" | cut -d. -f2)"
    if [[ "$HAVE_MAJOR" -gt "$MIN_MAJOR" ]] || \
       { [[ "$HAVE_MAJOR" -eq "$MIN_MAJOR" ]] && [[ "$HAVE_MINOR" -ge "$MIN_MINOR" ]]; }; then
        echo "  sbcl $HAVE_VERSION already installed (>= $SBCL_VERSION); skipping."
        sbcl --version
        exit 0
    fi
    echo "  sbcl $HAVE_VERSION found but < $SBCL_VERSION; reinstalling."
fi

# ─── apt deps (only what the binary tarball needs) ────────────────────
step "Installing apt prerequisites"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y -qq
    apt-get install -y -qq --no-install-recommends \
        ca-certificates curl wget bzip2 tar make
else
    echo "  apt-get not available; assuming curl/wget/bzip2 already on PATH"
fi

# ─── Fetch & extract ──────────────────────────────────────────────────
step "Fetching SBCL ${SBCL_VERSION} (${SBCL_ARCH}-linux)"
mkdir -p "$SBCL_TMPDIR"
cd "$SBCL_TMPDIR"

TARBALL="sbcl-${SBCL_VERSION}-${SBCL_ARCH}-linux-binary.tar.bz2"
URL="https://sourceforge.net/projects/sbcl/files/sbcl/${SBCL_VERSION}/${TARBALL}/download"

if [[ ! -f "$TARBALL" ]]; then
    # SourceForge serves a redirect, then a CDN mirror. -L follows it.
    curl -fL --retry 3 --retry-delay 5 -o "$TARBALL" "$URL" || {
        echo "  curl failed, trying wget fallback"
        wget --tries=3 -O "$TARBALL" "$URL"
    }
fi

# Sanity check: bzip2 archives have BZ magic bytes.
if ! head -c2 "$TARBALL" | grep -q '^BZ'; then
    echo "ERROR: downloaded file isn't a bzip2 archive — got:"
    head -c 200 "$TARBALL"
    exit 1
fi

step "Extracting"
rm -rf "sbcl-${SBCL_VERSION}-${SBCL_ARCH}-linux"
tar -xjf "$TARBALL"

# ─── Install ──────────────────────────────────────────────────────────
step "Installing into $INSTALL_ROOT"
cd "sbcl-${SBCL_VERSION}-${SBCL_ARCH}-linux"
INSTALL_ROOT="$INSTALL_ROOT" sh install.sh

# Make sure the freshly installed sbcl is what gets called below.
hash -r 2>/dev/null || true

# ─── Verify ───────────────────────────────────────────────────────────
step "Verifying"
if ! command -v sbcl >/dev/null 2>&1; then
    echo "ERROR: sbcl not on PATH after install. Check that $INSTALL_ROOT/bin is in PATH."
    exit 1
fi

sbcl --version
sbcl --non-interactive --eval \
    '(format t "~&__OK__sbcl ~A on ~A~%" (lisp-implementation-version) (machine-type))' \
    --eval '(sb-ext:exit)' | grep -q '__OK__' || {
    echo "ERROR: sbcl runs but a trivial --eval didn't produce expected output."
    exit 1
}

step "Done."
echo "  sbcl installed at: $(command -v sbcl)"
echo "  remove tmp:        rm -rf $SBCL_TMPDIR"
