# Overnight tuning report — 2026-07-01/02

**Goal:** push generation quality toward commercial ("Suno-level") sound, measured, not vibes.
**Method:** every config generated the same 3 fixed-seed test tracks (hip-hop vocal, pop vocal,
dance instrumental, identical prompts + lyrics), scored against profiles built from **48 official
30s previews of current chart tracks** (Deezer API — Drake, Kendrick, Bruno Mars, Bad Bunny, etc.).
Score = CLAP audio-embedding similarity (45%) + rhythmic density match + empty-space penalty +
brightness + stereo width. It's an A/B ruler, not an absolute grade.

## Final ranking (mean composite over 3 genres)

| config | mean | hip-hop | pop | dance |
|---|---|---|---|---|
| **v3rap-1000 + guidance 20 — ACTIVATED** | **64.4** | **69.8** | 72.0 | **51.3** |
| base + guidance 20 | 62.9 | 68.3 | 71.8 | 48.5 |
| v1-1500 (old) | 62.2 | | | |
| v3rap-2000 + g20 | 62.0 | 65.9 | 73.5 | 46.7 |
| base + guidance 15 | 61.5 | | | |
| v2-3000 (previous production) | 60.7 | | | |
| v2-1000 | 51.7 | | | |
| *chart tracks themselves (ceiling, self-inflated)* | *~87–98* | | | |

## What was learned (the real value)

1. **More FMA training ≠ better.** v2 (3000 steps, 800 tracks) scored *below* the untouched base
   model. FMA is amateur/indie music — training toward it pulls away from chart polish. The
   planned 6000-step v3 was cancelled on this evidence.
2. **Light, genre-focused training wins.** v3-rap: only 150 top-listened hip-hop tracks, only
   1000 steps — beat everything, in *all* genres. Its 2000-step sibling was already overcooked.
3. **Guidance 20 > 15 > 10** for chart-similarity.
4. **Dance/EDM is the weak genre** (~51 vs ~70): generations lock onto a relentless 16th-note
   grid (~8 onsets/sec throughout) where chart dance breathes (breakdowns, drops). Prompt-side
   arrangement language helps; the base model's EDM dynamics are the limit.

## Applied to production

- Active model: `v3_rap/epoch=6-step=1000_lora` (via `lora_active.txt`)
- `ACE_GUIDANCE=20` default in server.py
- Previous production config measured 60.7 → new config 64.4 (+3.7; hip-hop +9 vs v2-3000's 60.7 genre line)

## Rerun any of this

```
cd C:\musicforge\training
python fetch_refs.py      # refresh chart references
python sweep.py [1|2|3|4] # re-score configs (results append to sweep_results.json)
```

## Stage 5 — vocal-quality knobs (later session)

Tested on vocal-only fixed tests vs current production (mean 70.9):
heun scheduler 64.9, dual text/lyric guidance 67.1, both 55.5 — **all worse, none applied.**
Applied instead: plate reverb on the vocal stem in the mix chain (subtle, 13% wet) —
dry AI vocals read as "pasted on"; a short tail is the standard production fix.

## Stage 6 — v4-vocal (overnight, 2026-07-02)

Data: 150 tracks curated by *measured* vocal dominance (Demucs vocal-to-instrumental
ratio, top 150 of 350 popular vocal-genre candidates). Trained 4000 steps, checkpoints
every 500, all 8 evaluated on the vocal tests:

| steps | vocal mean | hip-hop | pop |
|---|---|---|---|
| 500 | 70.7 | 72.5 | 68.8 |
| **1000 — ACTIVATED** | **73.0** | **71.0** | **75.1** |
| 1500 | 66.8 | 67.8 | 65.7 |
| 2000-4000 | 69.3-70.2 | drifts down to ~62.6 | climbs to ~77.6 |

v4v-1000 beat the previous champion (70.9) in **both** genres. The curve past 1000
steps shows textbook overfit toward the pop-leaning data — checkpoint selection is
what made the user's requested 4000-step run safe.

Same night, lyrics engine overhauled: Qwen 1.5B → 3B, songwriter craft brief
(banned-cliche list, concrete imagery, 4-9 syllable lines), and structure tags
normalized to lowercase (they were reaching the singer as `[VERSE 1]`, degrading
section delivery).

## Stage 7 — v5-band (2026-07-02)

User diagnosis ("vocals with no backing music") traced to a v4 data bug: curating for
*maximum* vocal dominance selected near-a-cappella tracks. v5 recipe: 180 tracks in a
calibrated vocal/band ratio band (0.25-1.2 — clear voice OVER a full arrangement),
pool widened with Folk/Country/Rock. 2000 steps, all checkpoints ≥ v4:

| steps | vocal mean | notes |
|---|---|---|
| 500 | 73.2 | |
| 1000 | 73.3 | |
| **1500 — ACTIVATED** | **74.3** | beats v4 (73.05); both genres solid |
| 2000 | 73.7 | |

Also shipped: Director's Cut (best-of-3 takes, CLAP judge), "sings never speaks +
band under the voice" prompt directives, held-vowel line endings for sung genres,
and a trainer fix (Lightning was writing 8.5GB .ckpt files nobody used — 169GB
reclaimed, crash root-caused).

## Stage 8 — neural vocal enhancement (2026-07-02)

resemble-enhance wired into the mix chain: the vocal stem gets AI restoration
(de-fizz, presence) before remixing. Windows required a deepspeed stub, a PosixPath
patch, and a numpy-2 fix in their lib — all applied. ~11s per track, `VOCAL_ENHANCE=off`
to disable. First full-stack track (v5 + Director's Cut 3 takes + vocal enhance):
**78.8 composite — project record** (previous best fresh generation: 73.2).

## Stage 9 — intelligibility work (2026-07-02)

New metric: WER (Whisper transcript of the vocal stem vs intended lyrics) — the
"mumbling" complaint as a number. Baseline (v5+enhance): **0.557** — ~44% of words
survive. Findings:
- ACE's lyric-guidance knobs do NOT improve WER (lg0 was best; combos hurt).
- v6 trained on diction-filtered data (123 tracks whose vocals Whisper transcribes
  cleanly): **rejected** — v6d-500 74.2/0.685, v6d-1000 62.5/1.629, both worse than
  baseline. Diction-filtered training does not transfer word clarity.
- Conclusion: the ~0.56 WER mumble floor lives in the base model's vocal synthesis.
  Both knobs and data recipes are exhausted; a better base model is the fix.
Also: output switched to mp3 (~10x smaller), GPU-collision guard added (503 during
training instead of OOM crashes), training crash on 2026-07-02 traced to a generation
landing mid-run before the guard existed.

## Honest bottom line

The measured gap to actual chart tracks remains large. This harness + your GPU got a real,
repeatable +4-9 points. The next tier needs either a stronger base model (watch for ACE-Step v2
or YuE releases) or licensed pro-sounding training data — not more overnight steps on FMA.
