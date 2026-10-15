# AA-LCR retained evidence

Original activation command (do not replay):

```sh
cd <BENCHMARK_WORKSPACE>
python3 run_b2.py run --engine halo --cohort b2lc-20261015 --benchmark aa_lcr --ctx 163840 --slots 1 --timeout 259200
```

Full100-item cohort; output cap32768, xhigh, temperature1, top-p.95/top-k20/min-p0/base seed20261014. This export performs no generation or inference. Original judgment bytes are preserved; recovery is applied only when an explicit source-bound overlay is supplied.
