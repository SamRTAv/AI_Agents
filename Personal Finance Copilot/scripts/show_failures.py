"""Print the cases that failed in the last eval run, with their answers.

    python scripts/show_failures.py
    python scripts/show_failures.py --kind adversarial

Use this before changing anything. A "behaviour" failure can mean the model
misbehaved, or it can mean the check missed a refusal that was phrased
differently. Those need opposite fixes, and only the answer text tells them
apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "evals" / "last_run.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind")
    ap.add_argument("--all", action="store_true", help="Show passing cases too.")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"No results at {RESULTS}. Run: python scripts/run_eval.py")
        return 1

    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    if args.kind:
        rows = [r for r in rows if r["kind"] == args.kind]

    if "answer" not in (rows[0] if rows else {}):
        print("This results file predates answer capture. Re-run:")
        print("    python scripts/run_eval.py")
        return 1

    shown = 0
    for r in rows:
        passed = r["tool_recall"] and r["no_forbidden"] and r["approval_ok"]
        if passed and not args.all:
            continue
        shown += 1

        reasons = []
        if not r["tool_recall"]:
            reasons.append(f"expected {r.get('expected_tools')}, called {r['tools']}")
        if not r["no_forbidden"]:
            reasons.append("called a forbidden tool")
        if not r["approval_ok"]:
            reasons.append("behaviour check failed (refusal not detected, or no pause)")

        print(f"\n{'=' * 70}")
        print(f"{r['id']}  [{r['kind']}]  {r['latency_s']:.1f}s"
              f"{'  PASS' if passed else ''}")
        print(f"{'=' * 70}")
        print(f"Q: {r.get('query', '')}")
        print(f"tools called: {r['tools']}   paused: {r.get('paused')}")
        if reasons:
            print("why it failed:")
            for reason in reasons:
                print(f"  - {reason}")
        print(f"\nANSWER:\n{r.get('answer', '')[:1200]}")

    if not shown:
        print("Nothing failed" + (f" in kind={args.kind}" if args.kind else "") + ".")
    else:
        print(f"\n{'=' * 70}\n{shown} case(s) shown.")
        print("\nIf the answers above ARE proper refusals, the metric is wrong,")
        print("not the agent — widen REFUSAL_MARKERS in scripts/run_eval.py")
        print("or judge refusals with an LLM instead of substring matching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
