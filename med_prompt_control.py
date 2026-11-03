"""Control for med_prompt.py: the prompt lists 5 drugs that are NOT the one
spoken. Counts how often the transcript turns into one of the listed (wrong)
drugs, i.e. the prompt pulling the recognizer toward a medication the
patient didn't say."""
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
rng = random.Random(1)
rows = []
for model_name in ["tiny.en", "base.en", "small.en"]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    for cond in ["wide", "phone", "noisy"]:
        for it in manifest:
            right = it["answer"].split("|")[0]
            wrong = rng.sample([p for p in pool if p != right], 5)
            audio = score.read_wav(clip_dir / f"{it['stem']}_{cond}.wav")
            segs, _ = model.transcribe(audio, language="en", beam_size=1,
                                       initial_prompt="Patient medications: " + ", ".join(wrong) + ".")
            text = " ".join(s.text for s in segs).strip()
            got = score.norm_med(text)
            name = got.split("|")[0] if got else None
            rows.append({"model": model_name, "cond": cond, "answer": it["answer"], "got": got,
                         "correct": got == it["answer"], "swapped_to_listed": name in wrong, "text": text})
        sub = [r for r in rows if r["model"] == model_name and r["cond"] == cond]
        print(f"{model_name} {cond}: correct {sum(r['correct'] for r in sub)}/{len(sub)}, "
              f"swapped to a listed wrong drug {sum(r['swapped_to_listed'] for r in sub)}/{len(sub)}", flush=True)
Path("results_med_prompt_control.json").write_text(json.dumps(rows, indent=1))
