# phone-asr-entities

A small benchmark of the part of a patient phone call that a scheduling voice agent can't afford to get wrong: the date of birth, the callback number, the medication and dose, and a spelled last name. I wanted to know how open speech models (faster-whisper tiny/base/small, English) do on those entities once the audio goes through a phone line, and whether giving the recognizer the patient's medication list, which an agent with EHR access would already have, fixes drug names.

## Setup

- 96 utterances read by 4 Piper TTS voices (US x3, UK x1): 24 each of DOB, phone number, "refill on my <drug>, the <dose> milligram tablets", and spelled surname.
- Three audio conditions per clip: **wide** (16 kHz), **phone** (300-3400 Hz band, 8 kHz, 8-bit mu-law, back to 16 kHz), **noisy** (phone + low-passed noise at 10 dB SNR).
- faster-whisper on CPU, int8, beam size 1. A clip counts only if the entity comes out exactly right after normalization (digits for phone, ISO date for DOB, name and dose for meds, letters for spelling).

## Results

Exact entity match:

| model | audio | DOB | phone | med + dose | spelled name | all |
|---|---|---|---|---|---|---|
| tiny.en | wide | 100% | 92% | 12% | 58% | 66% |
| tiny.en | phone | 100% | 83% | 12% | 46% | 60% |
| tiny.en | noisy | 92% | 54% | 8% | 4% | 40% |
| base.en | wide | 100% | 100% | 17% | 67% | 71% |
| base.en | phone | 100% | 92% | 29% | 58% | 70% |
| base.en | noisy | 96% | 88% | 8% | 25% | 54% |
| small.en | wide | 96% | 100% | 42% | 83% | 80% |
| small.en | phone | 100% | 100% | 42% | 88% | 82% |
| small.en | noisy | 96% | 100% | 29% | 54% | 70% |

Median real-time factor on CPU: tiny 0.022, base 0.048, small 0.168.

Drug names are the weak spot by far. The dose was almost always right (1 wrong dose in the 195 medication transcripts my parser could read; 21 more, mostly noisy, didn't parse at all), but names came out as near-misses like "Tanzel Austin" (tamsulosin), "surgery line" (sertraline), "Gabby Penton" (gabapentin). Spelled surnames fall apart with noise.

**With the patient's medication list in the prompt** (`med_prompt.py`): each clip's prompt is "Patient medications: ..." with the right drug plus 4 others, shuffled.

| model | wide | phone | noisy |
|---|---|---|---|
| tiny.en | 12% -> 96% | 12% -> 83% | 8% -> 71% |
| base.en | 17% -> 88% | 29% -> 88% | 8% -> 79% |
| small.en | 42% -> 96% | 42% -> 92% | 29% -> 88% |

**Control** (`med_prompt_control.py`): the prompt lists 5 drugs that are *not* the one spoken. Across all 216 runs the transcript never turned into one of the listed wrong drugs, and accuracy stayed near the no-prompt baseline. So in this setup the list helps when it's right and didn't pull the model toward a wrong medication when it wasn't.

## Caveats

- TTS voices are cleaner and more regular than real callers, and Piper may mispronounce some drug names itself, so the med numbers mix TTS and recognition errors. Real calls would likely be worse.
- 24 clips per cell, so differences of one or two clips are noise.
- My 12-drug pool has fairly distinct names. Look-alike, sound-alike pairs (hydroxyzine / hydralazine) would be a much harder test of the control.
- One DOB miss is a scoring limitation: the model wrote the year as "Nineteen hundred fifty-two", which my normalizer doesn't parse.

## Run it

```
python -m venv venv && venv/bin/pip install faster-whisper piper-tts scipy pytest
# Piper voices go in ../voices (en_US-lessac/amy/ryan-medium, en_GB-alan-medium from rhasspy/piper-voices)
venv/bin/python make_clips.py          # writes clips/
venv/bin/python score.py clips         # results.json
venv/bin/python analyze.py results.json
venv/bin/python med_prompt.py clips && venv/bin/python med_prompt_control.py clips
venv/bin/python -m pytest -q
```
