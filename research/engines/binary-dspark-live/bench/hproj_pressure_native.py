#!/usr/bin/env python3
"""Expose the native FP4 body without instantiating its all-shape launch registry."""
from pathlib import Path
import json
import sys

if len(sys.argv) != 4 or sys.argv[3] not in ("native", "lanes"):
    raise SystemExit("usage: hproj_pressure_native.py SOURCE_FP4_CU OUTPUT_CUH native|lanes")
source, output = map(Path, sys.argv[1:3])
release = sys.argv[3]
text = source.read_text()
marker = "template <int FW, int WG, int TOKW, int MODE, int WF, int PIPE = 0, bool MICRO = false, bool CARRIER = false, bool DEGREE8 = false>\n__global__"
if text.count(marker) != 1:
    raise SystemExit("native FP4 wrapper boundary changed: refusing ambiguous extraction")
prefix = text[:text.index(marker)]
if prefix.count("void gemm_fp4_body(") != 1 or not prefix.rstrip().endswith("}"):
    raise SystemExit("native FP4 body boundary changed")
if release == "lanes":
    # Diagnostic alternative: each consuming lane publishes its own completed
    # reads, instead of aggregating them through syncwarp and a leader arrival.
    # Keep the same instruction order, storage, phases and number of stages.
    init = "mbar_init(&empty[s], FW);"
    arrive = "if (lane == 0) mbar_arrive(&empty[s]);"
    if prefix.count(init) != 1 or prefix.count(arrive) != 3:
        raise SystemExit("native empty-barrier protocol changed: refusing lane-release control")
    prefix = prefix.replace(init, "mbar_init(&empty[s], FW * 32);")
    prefix = prefix.replace(arrive, "mbar_arrive(&empty[s]);")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("#line 1 " + json.dumps(str(source.resolve())) + "\n" + prefix + "\n} // namespace h5\n")
