"""Singing-vs-talking detector. Talking = short, flat pitch blips. Singing =
held notes, wide contour. Measured on the Demucs vocal stem via pyin.

singing index 0-100 =
  60% sustain    (fraction of voiced time spent inside held notes >= 0.3s, +-60 cents)
  25% range      (p95-p5 pitch spread in semitones, credited between 2 and 12)
  15% voiced     (how much of the track has pitched voice at all)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "C:/musicforge")
import mix

SR = 22050
HOP = 256  # ~11.6ms frames
MIN_NOTE_S = 0.3
CENTS_TOL = 60


def vocal_stem(path):
    import librosa

    data, sr = sf.read(str(path))
    if data.ndim == 1:
        data = np.stack([data, data], 1)
    voc = mix._split(data, sr)["vocals"].mean(axis=1).astype("float32")
    return librosa.resample(voc, orig_sr=mix.STEM_SR, target_sr=SR)


def singing_metrics(path, stem=None):
    import librosa

    voc = vocal_stem(path) if stem is None else stem
    f0, voiced, _ = librosa.pyin(voc, fmin=80, fmax=800, sr=SR, hop_length=HOP)
    n = len(f0)
    v = voiced & ~np.isnan(f0)
    if v.sum() < 10:
        return {"singing": 0.0, "sustain_frac": 0.0, "range_st": 0.0,
                "voiced_frac": 0.0, "notes": 0, "longest_note_s": 0.0}
    cents = 1200 * np.log2(np.where(v, f0, np.nan) / 55.0)
    frame_s = HOP / SR
    min_frames = int(MIN_NOTE_S / frame_s)

    # sustained notes: voiced runs holding within +-CENTS_TOL of the run median
    notes, i = [], 0
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        run = cents[i:j]
        k = 0
        while k < len(run):
            m = k
            while m < len(run) and abs(run[m] - np.median(run[k:m + 1])) <= CENTS_TOL:
                m += 1
            if m - k >= min_frames:
                notes.append((m - k) * frame_s)
            k = max(m, k + 1)
        i = j

    voiced_frac = float(v.mean())
    sustain_frac = float(sum(notes) / (v.sum() * frame_s)) if v.sum() else 0.0
    vc = cents[v]
    range_st = float((np.percentile(vc, 95) - np.percentile(vc, 5)) / 100)
    singing = 100 * (
        0.60 * min(1.0, sustain_frac / 0.35)
        + 0.25 * min(1.0, max(0.0, range_st - 2) / 10)
        + 0.15 * min(1.0, voiced_frac / 0.5)
    )
    return {"singing": round(singing, 1), "sustain_frac": round(sustain_frac, 3),
            "range_st": round(range_st, 1), "voiced_frac": round(voiced_frac, 3),
            "notes": len(notes), "longest_note_s": round(max(notes), 2) if notes else 0.0}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(Path(f).name, singing_metrics(f))
