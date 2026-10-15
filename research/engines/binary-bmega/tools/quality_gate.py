#!/usr/bin/env python3
"""halo-5090 quality gate (GSM8K, HumanEval+, IFEval, MMLU-Redux) against a running OpenAI-compatible server.

  quality_gate.py --server URL --out DIR [--quick] [--resume]   run; writes DIR/summary.json and per-benchmark files
  quality_gate.py compare BASELINE_DIR CANDIDATE_DIR [--k 2] [--json]
                                                                PASS (exit 0) unless a gated metric drops by more than
                                                                k paired standard errors of the difference (FAIL, exit 1)
  quality_gate.py rescore DIR                                   re-score saved responses (no server)

Errors exit 2. Details: tools/QUALITY_GATE.md.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qgate import gate  # noqa: E402


def main(argv):
    if argv and argv[0] == "compare":
        p = argparse.ArgumentParser(prog="quality_gate.py compare")
        p.add_argument("baseline"); p.add_argument("candidate")
        p.add_argument("--k", type=float, default=2.0, help="standard errors of the paired difference (default 2)")
        p.add_argument("--json", action="store_true")
        return gate.compare(p.parse_args(argv[1:]))
    if argv and argv[0] == "rescore":
        p = argparse.ArgumentParser(prog="quality_gate.py rescore")
        p.add_argument("dir")
        return gate.rescore(p.parse_args(argv[1:]))
    p = argparse.ArgumentParser(prog="quality_gate.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true", help="GSM8K fresh32 + HumanEval+ only")
    p.add_argument("--benchmarks", nargs="+", choices=gate.FULL, help="explicit benchmark subset; selected suite, not a full gate")
    p.add_argument("--resume", action="store_true", help="reuse matching responses already in --out")
    p.add_argument("--limit", type=int, help="first N prompts per benchmark only (format probes on slow hosts, not a gate run)")
    p.add_argument("--concurrency", type=int, default=1, help="concurrent requests (batched accuracy runs; start the server with as many --slots)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 sends native sampling fields")
    p.add_argument("--top-p", type=float, default=0.95); p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.05); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draft-verify-limit", type=int, help="send draft_verify_limit (0 = no drafting)")
    return gate.run(p.parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except gate.GateError as e:
        print(f"quality_gate: error: {e}", file=sys.stderr)
        sys.exit(2)
