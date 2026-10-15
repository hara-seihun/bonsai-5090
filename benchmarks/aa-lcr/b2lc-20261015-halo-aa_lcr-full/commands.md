# AA-LCR original attempt

On one RTX 5090, using the retained runner and frozen manifest:

```sh
cd <BENCHMARK_WORKSPACE>
python3 run_b2.py run --engine halo --cohort b2lc-20261015 --benchmark aa_lcr --ctx 163840 --slots 1 --timeout 259200
```

The pinned configuration sets max_tokens=32768, thinking xhigh, temperature=1.0, top_p=0.95, top_k=20, min_p=0, seed=20261014 and Gemini 3 Flash judging at temperature 0. Credentials are supplied privately. The original run is never relaunched by an export. 60 generated was an earlier observation of this same 100-item attempt. All outputs and verdicts are retained; no regenerated answers or regrading in this package.
