# Pro engine (HeartMuLa)

The **Pro** toggle in MusicForge routes generation to [HeartMuLa](https://github.com/HeartMuLa/heartlib)
3B instead of the Basic ACE-Step pipeline — higher-quality vocals, run raw (no mix/master).

It runs as a subprocess in HeartMuLa's own venv (it needs `transformers==4.57`, incompatible with
ACE-Step's env). `server.py` frees ACE-Step from VRAM (`unload_ace`) before launching it, since both
need most of a 16GB card.

## Setup

```
git clone https://github.com/HeartMuLa/heartlib.git C:\heartlib
cd C:\heartlib
python -m venv .venv
.venv\Scripts\pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install numpy==2.0.2 transformers==4.57.0 tokenizers==0.22.1 accelerate==1.12.0 \
  einops==0.8.1 vector-quantize-pytorch==1.27.15 soundfile torchao>=0.16.0 \
  sentencepiece tiktoken
.venv\Scripts\pip install -e . --no-deps
# download checkpoints (~22GB) into C:\heartlib\ckpt per the heartlib README
copy <this repo>\pro\heartlib_gen.py C:\heartlib\heartlib_gen.py
```

Then set `HEARTLIB_DIR` if your clone isn't at `C:\heartlib`. `heartlib_gen.py` is the CLI the
server shells out to.

## Note on speed

HeartMuLa has no Triton kernels on Windows, so inference is slow (~8–15 min for a 60s track, more
for longer). On Linux/WSL with Triton it approaches real-time. The server gives Pro jobs a 90-minute
watchdog window accordingly.
