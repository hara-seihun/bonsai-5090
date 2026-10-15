#!/usr/bin/env python3
"""Produce disjoint content-pinned JSONL windows from one pinned benchmark split."""
import argparse,hashlib,json,random
from pathlib import Path
R=Path(__file__).resolve().parent

def retain(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==data,'Changed immutable shard '+str(path)
 else:path.write_bytes(data)

def tau2_windows(pin,size):
 source=(R/pin['snapshot_path']).resolve();task_name='tau2/domains/telecom/tasks.json';blobs={}
 for name,digest in pin['files_sha256'].items():
  data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest;blobs[name]=data
 records=json.loads(blobs[task_name]);assert len(records)==114
 indices=list(range(114));random.Random(20261014).shuffle(indices);directory=R/'tau2_bench-shards';windows=[]
 for i,start in enumerate(range(0,114,size)):
  selected=indices[start:start+size];d=directory/f'{i:03d}';hashes={}
  for name,data in blobs.items():
   p=d/name;p.parent.mkdir(parents=True,exist_ok=True)
   if name==task_name:
    data=(json.dumps([records[j]for j in selected],ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode();retain(p,data)
    assert json.loads(data)==[records[j]for j in selected]
   else:
    if p.exists()or p.is_symlink():assert p.is_symlink()and p.resolve()==(source/name).resolve()
    else:p.symlink_to(source/name)
   hashes[name]=hashlib.sha256(data).hexdigest()
  item={'benchmark':'tau2_bench','shard':i,'source_pin':pin,'source_sha256':pin['files_sha256'][task_name],'domains':['telecom'],'selected_original_indices':selected,'selected_ids':[records[j]['id']for j in selected],'files_sha256':hashes,'count':len(selected)}
  retain(d/'pin.json',(json.dumps(item,indent=2)+'\n').encode());windows.append({'shard':i,'count':len(selected),'indices':selected,'dataset_snapshot':str(d),'pin_sha256':hashlib.sha256((d/'pin.json').read_bytes()).hexdigest()})
 assert sorted(j for w in windows for j in w['indices'])==list(range(114))
 manifest={'benchmark':'tau2_bench','domains':['telecom'],'seed':20261014,'source_pin':pin,'source_sha256':pin['files_sha256'][task_name],'total':114,'window_size':size,'windows':windows,'pilot_shards':[0],'remaining_requires':'Five completed tasks costed and extrapolated against the remaining shared USD5 budget; no replay of pilot tasks'}
 retain(directory/'partition.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'benchmark':'tau2_bench','items':114,'windows':len(windows),'first_window_count':windows[0]['count'],'inference':False}));return

def main():
 layouts={
  'math_500':('test.jsonl','test.jsonl','test','unique_id'),
  'ifeval':('ifeval_input_data.jsonl','ifeval_input_data.jsonl','train','key'),
  'humaneval_plus':('test.jsonl','test.jsonl','test','task_id'),
  'mbpp_plus':('data/test-00000-of-00001-d5781c9c51e02795.parquet','test.jsonl','test','task_id'),
  'ifbench':('data/train-00000-of-00001.parquet','train.jsonl','train','key'),
  'tau2_bench':None,
 }
 p=argparse.ArgumentParser();p.add_argument('--benchmark',choices=list(layouts),required=True);p.add_argument('--size',type=int,required=True);a=p.parse_args()
 if a.size<=0:p.error('--size must be positive')
 lock=json.loads((R/'datasets-original15.lock.json').read_text());pin=lock[a.benchmark]
 if a.benchmark=='tau2_bench':return tau2_windows(pin,a.size)
 source_name,filename,split,id_key=layouts[a.benchmark]
 source=(R/pin['snapshot_path']).resolve()/source_name;source_bytes=source.read_bytes();digest=hashlib.sha256(source_bytes).hexdigest();assert digest==pin['files_sha256'][source_name]
 if source.suffix=='.parquet':
  import pyarrow.parquet as pq
  records=pq.read_table(source).to_pylist();lines=[(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n').encode()for x in records]
  assert all(json.loads(x)==record for x,record in zip(lines,records))
 else:
  lines=source_bytes.splitlines(keepends=True);records=[json.loads(x)for x in lines]
 assert len(records)==pin['declared_item_count']
 indices=list(range(len(records)));random.Random(20261014).shuffle(indices);directory=R/(a.benchmark+'-shards');directory.mkdir(exist_ok=True)
 windows=[]
 for i,start in enumerate(range(0,len(indices),a.size)):
  selected=indices[start:start+a.size];d=directory/f'{i:03d}';d.mkdir(exist_ok=True)
  body=b''.join(lines[j].rstrip(b'\r\n')+b'\n'for j in selected);retain(d/filename,body)
  card=f'---\nconfigs:\n- config_name: default\n  data_files:\n  - split: {split}\n    path: {filename}\n---\n'
  retain(d/'README.md',card.encode())
  item={'benchmark':a.benchmark,'shard':i,'source_pin':pin,'source_sha256':digest,'selected_original_indices':selected,'selected_ids':[records[j][id_key]for j in selected],'files_sha256':{filename:hashlib.sha256(body).hexdigest(),'README.md':hashlib.sha256(card.encode()).hexdigest()},'count':len(selected)}
  retain(d/'pin.json',(json.dumps(item,indent=2)+'\n').encode());windows.append({'shard':i,'count':len(selected),'indices':selected,'dataset_snapshot':str(d),'pin_sha256':hashlib.sha256((d/'pin.json').read_bytes()).hexdigest()})
 assert sorted(j for x in windows for j in x['indices'])==list(range(len(records)))
 manifest={'benchmark':a.benchmark,'seed':20261014,'source_sha256':digest,'source_pin':pin,'total':len(records),'window_size':a.size,'windows':windows,'overlapping_screens_excluded':True}
 retain(directory/'partition.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'benchmark':a.benchmark,'items':len(records),'windows':len(windows),'partition':str(directory/'partition.json'),'inference':False}))
if __name__=='__main__':main()
