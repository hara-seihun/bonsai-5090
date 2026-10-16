# CPU sampling contract

`-infinity` logits exclude tokens before penalties and filtering. NaN and positive infinity are errors. An empty permitted set returns an explicit error. Original token IDs, finite arithmetic, counter-based noise and lower-ID tie ordering remain unchanged.

The fixture invokes the actual `h5::sampling::select` through the same `qc_sample_cpu` wrapper and 48-byte configuration layout. It checks 10,368 masked configurations plus empty, invalid, tie and sole-survivor cases.

Run from the repository root (requires the system nlohmann-json development headers):

```sh
c++ -O2 -std=c++17 -Iengine/src engine/tests/sampling_masks.cpp -o /tmp/bonsai-sampling-masks
/tmp/bonsai-sampling-masks
```

This CPU boundary proof establishes no GPU execution, model quality, throughput or serving adoption.
