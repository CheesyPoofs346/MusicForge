# MusicForge training rig

Fine-tune the song model (ACE-Step 3.5B) with LoRA on your own audio, on your GPU.
This is real training — it changes how the model sounds. It is not training a new
model from scratch (nothing on one PC can); it's the strongest thing a 5080 can do.

## Quick start

```
cd C:\musicforge\training

# 1. Get training audio (either)
python get_free_data.py --limit 200 --genres "Hip-Hop,Electronic"   # free CC-licensed music (FMA), genre-filtered, auto-tagged
#   ...or drop your own .mp3/.wav files into training\raw\
#   (optional: add "<same name>.txt" next to a file with style tags)

# 2. Prep: transcribe lyrics (Whisper) + build the dataset
python prep.py                  # or:  python prep.py --instrumental

# 3. Train (defaults are tuned for 16GB VRAM; close the MusicForge server first!)
python train.py --max_steps 2000

# 4. Use it: point the server at the newest adapter and restart
#    training\exps\logs\musicforge_lora\<version>\checkpoints\epoch=..._lora
set ACE_LORA=C:\musicforge\training\exps\logs\musicforge_lora\version_0\checkpoints\epoch=0-step=2000_lora
```

## What to expect

- 100–500 well-tagged tracks in one consistent style beats 5000 random ones.
- ~2000 steps is a reasonable first run (hours, not minutes).
- Train on music you have the right to use: CC-licensed (what `get_free_data.py`
  fetches), public domain, or your own recordings — not ripped commercial tracks.
- The server holds ~8GB VRAM when loaded — stop it before training or you'll OOM.

## Files

- `get_free_data.py` — downloads FMA-small (CC-licensed ML dataset), stages tagged tracks
- `prep.py` — audio folder → lyrics (faster-whisper) + tags (sidecar or local LLM) → HF dataset
- `train.py` — LoRA fine-tune, 16GB-friendly defaults; all trainer flags pass through
- `trainer.py`, `lora_config.json` — vendored from the official ACE-Step repo
