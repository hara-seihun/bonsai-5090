from pathlib import Path
import json,time
# Execute only after the explicit PrismML handback, not during preparation.
d=Path('C:/work/models/experiments/bonsai-quality-20261013/fullk-h1-tail-v3')
p=d/'gpu-window-v3.json';now=time.time()
if p.exists():raise RuntimeError('existing window retained; name a successor rather than resetting its deadline')
record={'state':'explicit_handoff_window','started_epoch_s':now,'deadline_epoch_s':now+240,'candidate_dll_sha256':'398a9d2bb38ec8613af6663dd782a282727da04b5f255f43e4c10780f1c6d768'}
p.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
