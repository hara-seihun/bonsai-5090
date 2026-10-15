#!/bin/bash
set -euo pipefail
cd /workspace/halo-6000
exec python3 bench/loader-run.py
