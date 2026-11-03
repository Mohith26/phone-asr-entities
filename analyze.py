"""Summarize results.json: entity exact-match rate by model, condition and entity type."""
import json
import statistics
import sys
from collections import defaultdict

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
kinds = ["dob", "phone", "med", "spelled"]
acc = defaultdict(list)
lat = defaultdict(list)
for r in rows:
    acc[(r["model"], r["cond"], r["kind"])].append(r["ok"])
    acc[(r["model"], r["cond"], "all")].append(r["ok"])
    lat[r["model"]].append(r["secs"] / r["audio_secs"])

models = sorted({r["model"] for r in rows}, key=["tiny.en", "base.en", "small.en"].index)
print("| model | audio | " + " | ".join(kinds) + " | all |")
print("|---|---|" + "---|" * (len(kinds) + 1))
for m in models:
    for c in ["wide", "phone", "noisy"]:
        if (m, c, "all") not in acc:
            continue
        cells = [f"{100 * sum(acc[(m, c, k)]) / len(acc[(m, c, k)]):.0f}%" for k in kinds + ["all"]]
        print(f"| {m} | {c} | " + " | ".join(cells) + " |")
print()
for m in models:
    print(f"{m}: median real-time factor {statistics.median(lat[m]):.3f} (CPU, int8, beam 1)")
