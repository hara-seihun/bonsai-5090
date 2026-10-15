# One-plane absorption: rounding-scope correction

The sealed source768a66c0a/bundle f246296e and evidence archive5da515a8 are unchanged. Current projections retain the64 sampled-row coefficient-capacity obstruction and sparse actual-row witnesses.

`falsify.py` uses sequential binary32 RN; `rounding.cpp` explicitly selects FE_TONEAREST. Their positive1.5/K5120 construction is an **RN binary32 algebra witness**, not a native serving contraction falsifier. The reported native FP4 accumulation schedule matched RTZ; applying RTZ to that construction's final prefix gives12582912 in both arms. Native accumulation boundaries and rounding were not bound by the old packet. The generic2048-term construction also lacked serialized q4/s4/g4 layout fields, so its native-interface legality remains unset.

This correction does not alter the independent actual-row capacity result: the mean is not representable by one finite E2M1×UE4M3 coefficient, and the sparse constructed raw-operand witnesses expose that coefficient mismatch. Their scope remains interface operands on real fixed rows, not saved model activations or final-observer/TPS acceptance.

The richer joint-alphabet successor preserves the separate original gate/up operand words and native MMA instruction order. Its proof will treat the native contraction as an opaque operator, requiring identical operands at every call instead of assuming RN or RTZ behavior. A direct combined g+3u accumulator is not presumed sufficient for the separate rounded outputs or SiLU observer.
