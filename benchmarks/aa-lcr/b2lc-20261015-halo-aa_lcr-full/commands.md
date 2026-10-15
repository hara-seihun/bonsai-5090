# AA-LCR retained evidence

Original activation command (do not replay):

```sh
cd <BENCHMARK_WORKSPACE>
python3 run_b2.py run --engine halo --cohort b2lc-20261015 --benchmark aa_lcr --ctx 163840 --slots 1 --timeout 259200
```

Full100-item cohort; output cap32768, xhigh, temperature1, top-p.95/top-k20/min-p0/base seed20261014. This export performs no generation or inference. Original judgment bytes are preserved; recovery is applied only when an explicit source-bound overlay is supplied.

Continuation activation (already completed; do not replay):

```sh
python3 long-context-recovery/aalcr-original-resume/resume.py run --inference-authorized --allow-paid
```

CPU-only normalization: `python3 long-context-recovery/continuation-export/export.py --destination <FRESH_BUNDLE_DIRECTORY>`. This combines immutable sources; the original run files remain partial and unchanged. Index64 uses the distinct 8192-token/low judge recovery. Official answer5000, candidate4000; INCORRECT retained, including the judge explanation that the candidate followed rounding instructions. No GPU, model, paid call, submission or contact is performed by this exporter.
