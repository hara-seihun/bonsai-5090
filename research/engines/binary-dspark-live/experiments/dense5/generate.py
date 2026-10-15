#!/usr/bin/env python3
"""Compile the production compact reader and current control as two namespaces."""
from pathlib import Path
root = Path(__file__).resolve().parents[2]
production = (root / 'src/fp4.cu').read_text()
a = production.index('#define C(a, b, c) tab[i]')
b = production.index('    FP4_CFGS(C)', a)
# Keep only the paid discriminator's original FP16-scale store/atomic/GU modes.
reduced = production[:a] + '''#define C(a, b, c) tab[i][0][0] = fp4_fn<a, b, c, 0, 0>(); tab[i][2][0] = fp4_fn<a, b, c, 2, 0>(); tab[i][3][0] = fp4_fn<a, b, c, 3, 0>(); i++;
''' + production[b:]
control = reduced
candidate = reduced.replace('namespace h5 {', 'namespace c5 {\nusing namespace h5;')
# Expose WF3 through the same WF0 benchmark call signature. Everything else is identical.
candidate = candidate.replace('WF == 3', 'WF == 0')
candidate = '#include "kernels.cuh"\n#include "device.cuh"\n' + candidate
build = root / 'build/dense5'
build.mkdir(parents=True, exist_ok=True)
(build/'candidate.cu').write_text(candidate)
(build/'control.cu').write_text(control)
