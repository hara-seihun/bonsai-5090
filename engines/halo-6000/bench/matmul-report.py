#!/usr/bin/env python3
"""Summarize guarded native matmul prices, retaining exact/near-exact distinction."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ROWS=[1,8,32,128,512]
def main():
 cases={};counts={'weight_values':0,'K32_integer_partials':0,'exact_output_values':0,'subset_output_values':0,'compact_s1_partials':0,'compact_s1_output_comparisons':0};max_near=0;max_normalized=0
 for p in sorted((ROOT/'matmul-results').glob('*.jsonl')):
  records=[json.loads(l) for l in p.read_text().splitlines()]
  if records[-1].get('passed')!=True:raise RuntimeError(f'Incomplete case {p}')
  prices=[]
  for r in records:
   if r.get('guard')=='all_weight_bytes_and_original_scales':counts['weight_values']+=r['weights'];assert r['different']==0
   if r.get('guard')=='splitk_all_K32_integer_partials':counts['K32_integer_partials']+=r['partials'];assert r['different']==0
   if r.get('guard')=='splitk_scalar_bit_identity':counts['exact_output_values']+=r['outputs'];assert r['bit_differences']==0
   if r.get('guard','').startswith('nonzero_first_expert_'):counts['subset_output_values']+=r['outputs'];assert r['bit_differences']==0
   if r.get('guard')=='compact_s1_all_K32_integer_partials':counts['compact_s1_partials']+=r['partials'];assert r['different']==0
   if r.get('guard','').startswith('compact_s1_') and 'bit_differences' in r:counts['compact_s1_output_comparisons']+=r['outputs'];assert r['bit_differences']==0
   if r.get('guard')=='scalar_float_tree_difference':max_near=max(max_near,r['max_abs']);max_normalized=max(max_normalized,r['relative_to_max_output'])
  for columns in ROWS:
   cell=[r for r in records if r.get('columns')==columns]
   exact=min((r for r in cell if r.get('timing','').startswith(('mma_splitk_','mma_s1_compact_'))),key=lambda r:r['cold_us'])
   near=min((r for r in cell if r.get('timing')=='mma_q8'),key=lambda r:r['cold_us'])
   scalar=next(r for r in cell if r.get('timing')=='scalar_warp32')
   prep=next(r for r in cell if r.get('timing')=='prepared_B_only_no_quantizer')
   prices.append(dict(columns=columns,exact=exact,near=near,scalar32=scalar,prepared_B=prep))
  cases[p.stem]=prices
 result=dict(cases=cases,counts=counts,sequential_scalar_max_abs=max_near,sequential_scalar_max_normalized=max_normalized,scope='Real GGUF weights; synthetic already-quantized INT8/non-dyadic FP16-effective A32. Exact prices preserve scalar32 XOR tree. Quantization/rotation excluded; B packing separate. Resident188 means CTA grid-striding, no whole-model inference claim.')
 (ROOT/'matmul-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print('| Matrix / real source | s1 | s8 | s32 | s128 | s512 |')
 print('|---|---:|---:|---:|---:|---:|')
 for name,prices in cases.items():print('| '+name+' | '+' | '.join(f'{p["exact"]["cold_us"]:.2f}' for p in prices)+' |')
 print(json.dumps({**counts,'sequential_max_abs':max_near,'sequential_max_normalized':max_normalized}))
if __name__=='__main__':main()
