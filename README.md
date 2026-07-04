<div align="center">

# MusicForge

**Local AI music generation — full songs with sung lyrics, mixed and mastered, on your own GPU.**

No cloud, no credits, no limits. A model fine-tuned on this very machine.

</div>

---

## What it is

MusicForge is a self-hosted web app that generates complete ~60-second songs from a text prompt:
sung vocals with real lyrics (or instrumentals), stem-separated, balanced, and mastered to
streaming loudness — all running locally on a consumer GPU (built and tuned on a 16GB RTX 5080).

- **Type a prompt, pick a genre, choose Instrumental or Vocal** → get a finished, mastered track.
- **Sung lyrics** — auto-written by a local LLM (or Claude if an API key is present), or bring your own.
- **Style reference** — upload any track to steer the sound (audio2audio).
- **Director's Cut** — generate 3 takes and keep the one a CLAP judge scores closest to the brief.
- **Fullscreen player** — 4 live audio-reactive visualizers, spinning cover art, auto-scrolling lyrics.
- **Trainable** — fine-tune the model on your own audio with the included LoRA training rig.
- **Remote access** — one command exposes it over a Cloudflare tunnel for phone use.

## How a track is made

```
prompt ─▶ auto-enhance (LLM writes a dense, section-by-section prompt + lyrics + title)
       ─▶ ACE-Step 3.5B  (48kHz stereo generation, active LoRA fine-tune applied)
       ─▶ Demucs         (split into vocals / drums / bass / other)
       ─▶ per-stem mix   (vocal presence EQ + compression + plate reverb, sub cleanup, mud cut)
       ─▶ master         (30Hz HPF, normalize to -14 LUFS, soft limiter)
       ─▶ mp3
```

Everything downstream of generation is engine-agnostic and operates on the audio file, so the
generation model can be swapped without touching the mix/master chain.

## The model

The active voice is **MusicForge v7** — a LoRA fine-tune of [ACE-Step](https://github.com/ace-step/ACE-Step)
(3.5B), trained locally. It's the latest in a measured lineage where **every version had to beat the
previous one on a real evaluation harness** before being activated:

| eval axis | how it's measured |
|---|---|
| chart similarity | CLAP audio-embedding similarity to current chart-track previews |
| word clarity | Whisper transcript of the vocal stem vs. the intended lyrics (WER) |
| singing vs. talking | pyin pitch-tracking — sustained notes, pitch range, phrasing |

Key lesson from the lineage: **more training data isn't better**. Light, purpose-curated LoRAs won —
e.g. training on tracks selected for *measured melodic phrasing* (not just "vocal is loud") is what
finally made generations sing instead of talk. Full write-up in
[`training/tuning_report.md`](training/tuning_report.md).

## Run it

Requires Python 3.12 and a CUDA GPU. PyTorch (CUDA build) should be installed separately.

```bash
pip install -r requirements.txt          # torch installed separately (CUDA build)
python server.py                          # serves on http://127.0.0.1:8137
```

First run downloads the ACE-Step checkpoint (~3.5GB). Set `ANTHROPIC_API_KEY` to use Claude for
lyrics instead of the local LLM. `start.bat` launches the server plus a Cloudflare tunnel for
remote/phone access.

Tunable via env vars: `ACE_STEPS` (diffusion steps), `ACE_GUIDANCE`, `TRACK_SECONDS`,
`MIX=off`, `VOCAL_ENHANCE=off`, `ACE_LORA` (path to a checkpoint).

## Train your own

```bash
cd training
python get_free_data.py --limit 200 --genres "Hip-Hop,Pop"   # CC-licensed audio (FMA), auto-tagged
python prep.py                                                # Whisper lyrics + dataset build
python train.py --max_steps 2000                             # LoRA fine-tune (stop the server first)
# point lora_active.txt at the new checkpoint and restart
```

Train only on audio you have the right to use — Creative Commons, public domain, or your own.
See [`training/README.md`](training/README.md).

## Layout

```
server.py            FastAPI backend: generation queue, mix/master, API, access gating
mix.py               Demucs stem-split + per-stem treatment + rebalance
static/index.html    Single-file web UI (vanilla JS, glassmorphism, canvas visualizers)
training/            Curation, prep, LoRA trainer, and the CLAP/WER/pitch eval harness
models/              Published LoRA checkpoint (Git LFS)
test_smoke.py        API smoke test (mocks the GPU)
```

## Honest limits

- Vocal timbre still reads slightly synthetic — that's the base model's ceiling; a LoRA nudges style,
  not the synthesis engine. The eval harness is ready to benchmark a stronger base model (ACE-Step v2,
  YuE, etc.) the day one ships.
- Generation is heavy (~3–8 min/track depending on mode and GPU contention). Closing other
  GPU-hungry apps speeds it up materially.

## License & credits

App code is provided as-is. Built on open-source models — ACE-Step (Apache-2.0), Demucs, faster-whisper,
resemble-enhance, CLAP — each under its own license. Training uses only Creative-Commons audio.