# AA-LCR output-cap 40,960 — PARTIAL interrupted cohort

This separate 100-case cohort has **12 retained generations, 10 correct valid reviews,
2 excluded reviews and 88 absent generation responses**. The original dispatch of those
88 cases is unknown. **The full-cohort score is unset (`null`).** No accuracy percentage
is reported from this partial cohort.

The output cap is 40,960 tokens, context 163,840, one stream, temperature 1.0,
top-p 0.95, top-k 20 and seed 20261014. [Settings](settings.json) bind the model,
runtime, template and original source hashes. [Result](result.json) retains the counts
and index partitions. [Outputs](outputs.jsonl.gz) has all 100 explicit case identities:
answers, reasoning and raw output for indices 0–11, and absent-response states for 12–99.

Review index 1 is excluded because its retained judge response failed parsing;
index 2 is excluded after a judge transport failure with the paid outcome unknown and
no retained cache. Neither was rejudged. The run was interrupted for small-model speed
experiments; its terminal return code was 1. Missing-only continuation was not activated
for this export. Full rerun, needle retrieval and short-chat benchmarks remain open.

The original output-cap 32,768 **73/100** result remains separate and unchanged in
[the completed cohort](../b2lc-20261015-halo-aa_lcr-full/).
This publication performed no inference, judging or replay. Machine process identifiers
and private stop-reason wording are omitted; the retained output archive is byte-identical
to the accepted export. [Manifest](manifest.json) binds original and public file hashes.
