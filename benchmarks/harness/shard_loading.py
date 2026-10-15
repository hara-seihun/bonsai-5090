"""Owner-local, content-bound shard loading without basename-keyed HF cache reuse."""
import hashlib
import json
from pathlib import Path

class ShardLoadingFailure(RuntimeError):
    """Pinned local data cannot be loaded as the declared original task window."""

def install(snapshot):
    from evalscope.api.dataset.loader import LocalDataLoader, RemoteDataLoader
    root=Path(snapshot).resolve()
    pin=json.loads((root/'pin.json').read_text())
    files=pin['files_sha256']
    for name,digest in files.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ShardLoadingFailure('Shard file identity mismatch: '+name)
    payloads=[name for name in files if name.endswith('.jsonl')]
    if len(payloads)!=1:raise ShardLoadingFailure('Expected exactly one pinned JSONL payload')
    payload=(root/payloads[0]).resolve()
    rows=[json.loads(line)for line in payload.read_bytes().splitlines()]
    key={'humaneval_plus':'task_id','mbpp_plus':'task_id','ifbench':'key'}[pin['benchmark']]
    if len(rows)!=pin['count'] or [x[key]for x in rows]!=pin['selected_ids']:
        raise ShardLoadingFailure('Shard count or source task identities differ')
    split={'humaneval_plus':'test','mbpp_plus':'test','ifbench':'train'}[pin['benchmark']]
    official=RemoteDataLoader.load
    def load(self):
        if Path(self.data_id_or_path).resolve()!=root:return official(self)
        if self.subset!='default' or self.split!=split:
            raise ShardLoadingFailure('Undeclared subset/split for pinned shard')
        loader=LocalDataLoader(data_id_or_path=str(payload),split=self.split,
            sample_fields=self.sample_fields,filter_func=self.filter_func,subset=self.subset,
            version=self.version,limit=self.limit,data_source=self.data_source,
            shuffle=self.shuffle,shuffle_choices=self.shuffle_choices,seed=self.seed,
            auto_id=self.auto_id,repeats=self.repeats,trust_remote=self.trust_remote,
            force_redownload=self.force_redownload,dataset_dir=self.dataset_dir,**self.kwargs)
        return loader.load()
    RemoteDataLoader.load=load
