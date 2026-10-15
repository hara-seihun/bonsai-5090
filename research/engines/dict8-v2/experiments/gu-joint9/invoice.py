from pathlib import Path
import json,re,hashlib
p=Path(__file__).parent;root=Path('~/work/five-wide-native/gu-joint9-compile-v1')
s=(root/'probe.sass').read_text();out={}
for sec in s.split('Function : ')[1:]:
 name=sec.splitlines()[0];ins=[]
 for m in re.finditer(r'/\*([0-9a-fA-F]+)\*/\s+([^;]+);',sec):ins.append((int(m[1],16),m[2].strip()))
 loops=[]
 for pc,op in ins:
  m=re.search(r'BRA(?:\.\w+)*\s+0x([0-9a-fA-F]+)',op)
  if m and int(m[1],16)<pc:
   lo=int(m[1],16);body=[o for a,o in ins if lo<=a<=pc]
   omma=sum('OMMA.' in o for o in body)
   if omma:loops.append({'from':hex(lo),'backedge':hex(pc),'instruction_lines':len(body),'omma':omma,'lds':sum('LDS' in o for o in body),'idx':sum('SHFL.IDX' in o for o in body)})
 out[name]={'static_instruction_lines':len(ins),'static_omma':sum('OMMA.' in o for _,o in ins),'static_lds':sum('LDS' in o for _,o in ins),'backedge_regions_with_omma':loops}
record={'gpu_work':False,'native_model_execution':False,'matched_compiler_receipt_sha256':hashlib.sha256((root/'receipt.json').read_bytes()).hexdigest(),'functions':out,'whole_gu_route':{'B':64,'K':5120,'logical_output_features':17408,'physical_gate_up_rows':34816,'cta_grid':[2,68,1],'consumer_warps_per_cta':8,'K128_stages':40,'MMA_half_K':64,'MMA_sites_per_warp_stage':32,'warp_MMA_instructions_per_layer_decode':136*8*40*32,'WF0_weight_bytes_per_layer':1088*40*1024,'joint9_weight_bytes_per_layer':544*40*1664,'weight_byte_delta_fraction':-0.1875,'prepared_weight_bytes_64_layers':544*40*1664*64,'row_mean_bytes_per_layer_unchanged':34816*4,'raw_GU_SiLU_output_bytes_per_layer_unchanged':64*17408*4,'minimum_GU_write_plus_one_observer_read_bytes_per_layer_unchanged':64*17408*8,'additional_joint_table_bytes_per_CTA':2048,'additional_joint_tables_per_layer':136,'MMA_instruction_count_delta':0,'activation_q4_s4_g4_format_delta':0,'scale_AG_mean_order_delta':0,'downstream_SiLU_sign_Hadamard1024_FP4_observer_delta':0,'non_GU_projection_delta':0,'global_GU_carrier_newly_introduced':False},'unmeasured':['executed ALU/LDS/producer branch instruction counts','bank conflicts','latency share','registered occupancy','linked full engine/capture','full-model endpoints and TPS','absolute complete-observer traffic; original consumer retained and its delta is zero'],'scope':'Payload and source-schedule accounting; compiler static backedge regions are not cycle or executed-work measurements. This probe leaves the original complete consumer unchanged; no H1 carrier is introduced.'}
(p/'cost-invoice-v1.json').write_text(json.dumps(record,indent=2));print(json.dumps(out,indent=2))
