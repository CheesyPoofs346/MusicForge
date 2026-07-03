"""Curate a singing-focused training set: take the most-listened tracks from
vocal genres, Demucs-split each one, keep the ~150 with the highest measured
vocal-to-instrumental ratio. Voice-forward data for a voice-forward LoRA."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, "C:/musicforge")
import mix

HERE = Path(__file__).parent
FMA = HERE / "fma"
RAW = HERE / "raw"

CANDIDATES = 400
KEEP = 150
# v7: melody-only pool. Rap is spoken-cadence delivery — it was teaching the
# model to talk. Singing genres only.
GENRES = {"Pop", "Soul-RnB", "Folk", "Country", "Rock"}
# v4 lesson: maximizing vocal ratio selects a-cappella-ish tracks and the model
# forgets the band. v5 wants BOTH: clear vocals AND a full arrangement under them.
# Calibration: in real mixes vocals RMS ≈ 0.3-0.7x the summed instrumental stems.
RATIO_BAND = (0.25, 1.2)

tracks = pd.read_csv(FMA / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
pool = tracks[[("track", "genre_top"), ("track", "listens")]].copy()
pool.columns = ["genre", "listens"]
pool = pool[pool.genre.isin(GENRES)].sort_values("listens", ascending=False)

rms = lambda x: float(np.sqrt((x ** 2).mean()) + 1e-9)

import singing as singing_mod

# chart phrasing envelope (measured from actual chart vocals): held notes exist
# but are short and human — not the 5s drones our gens produce
ENVELOPE = {"sustain": (0.18, 0.55), "range_st": (5, 30), "longest": (0.0, 2.0), "singing": 55}


def phrasing(vocals):
    """Full singing metrics on the vocal stem — curated to the chart envelope."""
    import librosa

    stem = librosa.resample(vocals.mean(axis=1).astype("float32"),
                            orig_sr=mix.STEM_SR, target_sr=singing_mod.SR)
    return singing_mod.singing_metrics(None, stem=stem)


scored = []
checked = 0
for tid, row in pool.iterrows():
    if checked >= CANDIDATES:
        break
    src = FMA / "fma_medium" / f"{tid:06d}"[:3] / f"{tid:06d}.mp3"
    if not src.exists():
        continue
    try:
        data, sr = sf.read(src)
        if data.ndim == 1:
            data = np.stack([data, data], 1)
        stems = mix._split(data, sr)
    except Exception:
        continue
    ratio = rms(stems["vocals"]) / rms(stems["drums"] + stems["bass"] + stems["other"])
    if RATIO_BAND[0] <= ratio <= RATIO_BAND[1]:
        m = phrasing(stems["vocals"])
        if (m["singing"] >= ENVELOPE["singing"]
                and ENVELOPE["sustain"][0] <= m["sustain_frac"] <= ENVELOPE["sustain"][1]
                and ENVELOPE["range_st"][0] <= m["range_st"] <= ENVELOPE["range_st"][1]
                and m["longest_note_s"] <= ENVELOPE["longest"][1]):
            scored.append((m["singing"], src, str(row.genre)))
    checked += 1
    if checked % 50 == 0:
        print(f"scored {checked}/{CANDIDATES}, kept so far {len(scored)}", flush=True)

scored.sort(reverse=True)  # strongest in-envelope singers first

RAW.mkdir(exist_ok=True)
for ratio, src, genre in scored[:KEEP]:  # most-listened first (pool order preserved)
    shutil.copy(src, RAW / src.name)
    (RAW / f"{src.stem}.txt").write_text(
        f"{genre}, melodic sustained singing, held notes, expressive lead vocal over a full band", encoding="utf-8"
    )
print(f"kept {min(KEEP, len(scored))} band-balanced vocal tracks of {checked} checked "
      f"(ratio band {RATIO_BAND[0]}-{RATIO_BAND[1]})")
