"""Transcribe every clip with faster-whisper and check whether the entity a
scheduling agent needs (DOB, callback number, medication + dose, spelled
surname) survives exactly."""

import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
WORD_DIGITS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
               "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def norm_dob(t):
    t = t.lower()
    m = re.search(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t)
    if m:
        return f"{int(m.group(3)):04d}-{MONTHS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def norm_phone(t):
    t = t.lower()
    for w, d in WORD_DIGITS.items():
        t = re.sub(rf"\b{w}\b", d, t)
    tail = t.split("reach me is")[-1] if "reach me is" in t else t
    digits = re.sub(r"\D", "", tail)
    return digits or None


def norm_med(t):
    t = t.lower()
    m = re.search(r"refill on (?:my )?([a-z\- ]+?),?\s+(?:the\s+)?(\d+)\s*(?:milligram|mg)", t)
    if not m:
        return None
    return f"{m.group(1).strip().replace(' ', '').replace('-', '')}|{m.group(2)}"


def norm_spelled(t):
    t = t.lower()
    tail = t.split("that's")[-1] if "that's" in t else t.split("that is")[-1]
    letters = re.findall(r"\b([a-z])\b", tail)
    return "".join(letters) or None


NORM = {"dob": norm_dob, "phone": norm_phone, "med": norm_med, "spelled": norm_spelled}


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main(clip_dir="clips", models=("tiny.en", "base.en", "small.en"), out="results.json"):
    from faster_whisper import WhisperModel

    manifest = json.loads((Path(clip_dir) / "manifest.json").read_text())
    rows = []
    for model_name in models:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        for cond in ["wide", "phone", "noisy"]:
            for it in manifest:
                audio = read_wav(Path(clip_dir) / f"{it['stem']}_{cond}.wav")
                t0 = time.perf_counter()
                segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
                text = " ".join(s.text for s in segs).strip()
                dt = time.perf_counter() - t0
                got = NORM[it["kind"]](text)
                rows.append({"model": model_name, "cond": cond, "voice": it["voice"], "kind": it["kind"],
                             "answer": it["answer"], "got": got, "ok": got == it["answer"],
                             "text": text, "secs": round(dt, 3), "audio_secs": round(len(audio) / 16000, 2)})
            print(model_name, cond, "done", flush=True)
            Path(out).write_text(json.dumps(rows, indent=1))
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["clips"]))
