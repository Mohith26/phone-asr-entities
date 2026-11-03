"""Generate scheduling-call utterances with Piper TTS, in three audio conditions.

wide:  16 kHz PCM straight from the TTS voice
phone: 300-3400 Hz band, 8 kHz, 8-bit mu-law (what a PSTN call carries), back to 16 kHz
noisy: phone + background noise at 10 dB SNR
"""

import json
import random
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice
from scipy.signal import butter, resample_poly, sosfilt

VOICES = ["en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-medium", "en_GB-alan-medium"]
VOICE_DIR = Path(__file__).resolve().parent.parent / "voices"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MEDS = [("atorvastatin", 20), ("levothyroxine", 50), ("metformin", 500),
        ("gabapentin", 300), ("lisinopril", 10), ("amlodipine", 5),
        ("sertraline", 50), ("losartan", 25), ("omeprazole", 20),
        ("hydrochlorothiazide", 25), ("meloxicam", 15), ("tamsulosin", 4)]
SURNAMES = ["Nguyen", "Kowalczyk", "Okafor", "Patel", "Schreiber", "Mbeki",
            "Delacroix", "Haugen", "Ramaswamy", "Fitzgerald", "Ibarra", "Lindqvist"]

ORD = {1: "1st", 2: "2nd", 3: "3rd", 21: "21st", 22: "22nd", 23: "23rd", 31: "31st"}


def items(rng):
    out = []
    for _ in range(6):
        m, d, y = rng.randrange(12), rng.randint(1, 28), rng.randint(1940, 2010)
        out.append({"kind": "dob",
                    "text": f"Hi, I'm calling to book an appointment. My date of birth is {MONTHS[m]} {ORD.get(d, f'{d}th')}, {y}.",
                    "answer": f"{y:04d}-{m + 1:02d}-{d:02d}"})
    for _ in range(6):
        digits = f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"
        spoken = f"{digits[:3]}, {digits[3:6]}, {digits[6:]}"
        out.append({"kind": "phone",
                    "text": f"Sure, the best number to reach me is {' '.join(spoken)}.",
                    "answer": digits})
    for med, dose in rng.sample(MEDS, 6):
        out.append({"kind": "med",
                    "text": f"I need a refill on my {med}, the {dose} milligram tablets.",
                    "answer": f"{med}|{dose}"})
    for name in rng.sample(SURNAMES, 6):
        out.append({"kind": "spelled",
                    "text": f"My last name is {name}, that's {', '.join(name.upper())}.",
                    "answer": name.lower()})
    return out


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, x, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def mulaw(x, mu=255.0):
    """8-bit G.711-style mu-law round trip."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu


def to_phone(x16):
    sos = butter(6, [300, 3400], btype="bandpass", fs=16000, output="sos")
    x8 = resample_poly(sosfilt(sos, x16), 1, 2)
    x8 = mulaw(np.clip(x8, -1, 1))
    return resample_poly(x8, 2, 1).astype(np.float32)


def main(out_dir="clips", seed=0):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    manifest = []
    for voice in VOICES:
        tts = PiperVoice.load(str(VOICE_DIR / f"{voice}.onnx"))
        for i, it in enumerate(items(rng)):
            stem = out / f"{voice}_{i:02d}"
            chunks = list(tts.synthesize(it["text"]))
            x = np.concatenate([c.audio_float_array for c in chunks])
            sr = chunks[0].sample_rate
            g = np.gcd(16000, sr)
            x16 = resample_poly(x, 16000 // g, sr // g).astype(np.float32) * 0.8
            write_wav(Path(f"{stem}_wide.wav"), x16)
            ph = to_phone(x16)
            write_wav(Path(f"{stem}_phone.wav"), ph)
            noise = sosfilt(butter(1, 1000, fs=16000, output="sos"), nrng.normal(0, 1, len(ph)))
            p_sig = np.mean(ph[np.abs(ph) > 1e-3] ** 2)
            noise *= np.sqrt(p_sig / 10 ** (10 / 10) / np.mean(noise ** 2))
            write_wav(Path(f"{stem}_noisy.wav"), ph + noise)
            manifest.append({**it, "voice": voice, "stem": stem.name})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "utterances x 3 conditions")


if __name__ == "__main__":
    main()
