"""Does giving the recognizer the patient's medication list (as a scheduling
agent with EHR access could) fix drug-name errors?

For each medication clip, the prompt lists the right drug plus 4 other drugs
from the pool (a plausible 5-drug med list), shuffled. Same clips, same
models, beam 1, so the only change is the prompt."""
import json
import random
import sys
from pathlib import Path

from faster_whisper import WhisperModel

import score
from make_clips import MEDS

clip_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "clips")
manifest = [m for m in json.loads((clip_dir / "manifest.json").read_text()) if m["kind"] == "med"]
pool = [m for m, _ in MEDS]
rng = random.Random(0)
med_lists = {}
for it in manifest:
    right = it["answer"].split("|")[0]
    meds = [right] + rng.sample([p for p in pool if p != right], 4)
    rng.shuffle(meds)
    med_lists[it["stem"]] = meds

rows = []
for model_name in ["tiny.en", "base.en", "small.en"]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    for cond in ["wide", "phone", "noisy"]:
        for it in manifest:
            audio = score.read_wav(clip_dir / f"{it['stem']}_{cond}.wav")
            prompt = "Patient medications: " + ", ".join(med_lists[it["stem"]]) + "."
            segs, _ = model.transcribe(audio, language="en", beam_size=1, initial_prompt=prompt)
            text = " ".join(s.text for s in segs).strip()
            got = score.norm_med(text)
            rows.append({"model": model_name, "cond": cond, "answer": it["answer"], "got": got,
                         "ok": got == it["answer"], "text": text, "prompt": prompt})
        n = [r["ok"] for r in rows if r["model"] == model_name and r["cond"] == cond]
        print(f"{model_name} {cond}: {100 * sum(n) / len(n):.0f}% ({sum(n)}/{len(n)})", flush=True)
Path("results_med_prompt.json").write_text(json.dumps(rows, indent=1))
