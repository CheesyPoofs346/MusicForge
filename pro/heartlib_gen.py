"""HeartMuLa generation CLI — called as a subprocess by MusicForge's server
(runs in this repo's isolated venv, which has transformers 4.57 etc.).

Usage:
  python heartlib_gen.py --tags-file T.txt --lyrics-file L.txt --seconds 120 --out out.mp3

--tags / --lyrics may be raw strings or file paths (HeartMuLa reads a file if the
value is a path). The server passes temp-file paths to avoid CLI escaping issues.
Output is written at 48kHz and peak-normalized to -1 dBFS (clip safety only — no
EQ / compression / mixing, so this is 'just their model').
"""
import argparse
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio


# torchaudio.save routes through torchcodec on this version (broken on Windows); soundfile
def _sf_save(uri, src, sample_rate, **kw):
    sf.write(str(uri), src.detach().cpu().float().numpy().T, sample_rate)


torchaudio.save = _sf_save

from heartlib import HeartMuLaGenPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--tags-file", default="")
    ap.add_argument("--lyrics", default="[instrumental]")
    ap.add_argument("--lyrics-file", default="")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=r"C:\heartlib\ckpt")
    a = ap.parse_args()

    tags = a.tags_file or a.tags            # HeartMuLa reads a file if given a path
    lyrics = a.lyrics_file or a.lyrics

    t0 = time.time()
    print(f"loading HeartMuLa 3B...", flush=True)
    pipe = HeartMuLaGenPipeline.from_pretrained(
        a.ckpt,
        device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B", lazy_load=False,
    )
    print(f"loaded in {time.time()-t0:.0f}s, generating {a.seconds}s...", flush=True)

    t0 = time.time()
    with torch.no_grad():
        pipe(
            {"lyrics": lyrics, "tags": tags},
            max_audio_length_ms=a.seconds * 1000,
            save_path=a.out,
            topk=50, temperature=1.0, cfg_scale=1.5,
        )
    print(f"generated in {time.time()-t0:.0f}s", flush=True)

    # clip safety only (no mixing): peak-normalize to -1 dBFS
    data, sr = sf.read(a.out)
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.891  # -1 dBFS
    sf.write(a.out, data, sr)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
