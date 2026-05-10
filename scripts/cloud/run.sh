#!/bin/bash
set -euo pipefail
pip install -r scripts/cloud/requirements.txt -q
accelerate launch scripts/cloud/train_sft.py
