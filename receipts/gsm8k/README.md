# GSM8K: 1276 / 1319

`outputs.jsonl.gz` holds one line per test item: dataset index, request ID, seed, the full generated
text (reasoning plus final answer), finish reason and token usage. These are the outputs of the
measured run.

| | |
|---|---|
| Engine | the same `quality_custom` binary as the throughput receipts (sha256 `a836fd68…`) |
| Server | `engine/serve.py`, 8 slots, context 20,480, `profiles/serving.json` with 8 graph rows and 8 FP4 KV slots |
| Dataset | `openai/gsm8k`, `main`, `test`, revision `740312add88f781978c0658806c59bc2815b9866` (parquet sha256 `ee7b8da9…`), all 1,319 items |
| Prompt | GGUF chat template (template bytes sha256 `c3cf9e34…`), system prompt "Reasoning effort is set to xhigh. …", thinking on; exact bytes in `scripts/gsm8k.py` |
| Sampling | temperature 0.7, top-p 0.95, top-k 20, min-p 0; max 16,384 new tokens; one sample per item; per-item seed from campaign seed 20261014 |
| Result | **1276 correct / 1319 (96.74%)**; 9 items hit the 16,384-token cap without closing their reasoning and count as wrong |

**Scoring.** The original run used EvalScope's GSM8K adapter, whose answer extraction and comparison is
Math-Verify; no LLM judge was invoked for any item. `scripts/gsm8k.py --score-only` reimplements that
rule (text after `</think>`, Math-Verify parse and verify against the `####` answer). On these outputs
it agrees with the original per-item verdicts on all 1,319 items.

**Request identity.** For all 1,319 items, the request body `scripts/gsm8k.py` sends (prompt, sampling,
seed, stop strings) is byte-identical to the request recorded during the measured run.

For reference, PrismML's published GSM8K figure for Ternary Bonsai 2 27B is 96.06%. It was measured with
PrismML's own setup, not this protocol, so the two aren't a controlled comparison.

Rescore:

```bash
mkdir -p results/gsm8k-receipt && zcat receipts/gsm8k/outputs.jsonl.gz > results/gsm8k-receipt/outputs.jsonl
python scripts/gsm8k.py --out results/gsm8k-receipt --score-only
```
