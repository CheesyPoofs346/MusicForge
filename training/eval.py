"""Score generated tracks against chart-track reference profiles (see fetch_refs.py).

Measures what "sounds pro" actually decomposes into:
  gaps      fraction of near-empty time (the "empty spots" complaint)
  onset     rhythmic density, onsets/sec
  centroid  spectral brightness
  width     stereo width
  crest     punch (peak-to-RMS)
  clap_sim  CLAP audio-embedding similarity to the genre's chart tracks (machine ears)
Composite = weighted closeness to the genre profile, 0-100. Only comparable
between runs of the same harness — it's an A/B ruler, not an absolute grade.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

HERE = Path(__file__).parent
REFS = HERE / "refs"

_clap = None
_profiles = {}


def metrics(path):
    import librosa

    y, sr = librosa.load(str(path), sr=44100, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    db = librosa.amplitude_to_db(rms + 1e-8)
    gaps = float((db < db.max() - 30).mean())
    onset = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    y2, _ = sf.read(str(path))
    width = 0.0 if y2.ndim == 1 else float(1 - np.corrcoef(y2[:, 0], y2[:, 1])[0, 1])
    crest = float(np.abs(y).max() / (np.sqrt((y ** 2).mean()) + 1e-9))
    return dict(gaps=gaps, onset=onset, centroid=cent, width=width, crest=crest)


def clap_embed(path):
    global _clap
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    if _clap is None:
        m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda").eval()
        p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _clap = (m, p)
    m, p = _clap
    y, _ = librosa.load(str(path), sr=48000, mono=True)
    win = 10 * 48000
    chunks = [y[i:i + win] for i in range(0, max(1, len(y) - win + 1), win)][:6]
    ins = p(audios=chunks, sampling_rate=48000, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        e = m.get_audio_features(**ins)
    e = torch.nn.functional.normalize(e, dim=-1).mean(0)
    return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


def genre_profile(genre):
    if genre not in _profiles:
        files = sorted((REFS / genre).glob("*.mp3"))
        ms = [metrics(f) for f in files]
        prof = {k: float(np.median([m[k] for m in ms])) for k in ms[0]}
        emb = np.stack([clap_embed(f) for f in files]).mean(0)
        _profiles[genre] = (prof, emb / np.linalg.norm(emb))
    return _profiles[genre]


def score(path, genre):
    prof, ref_emb = genre_profile(genre)
    m = metrics(path)
    clap_sim = float(clap_embed(path) @ ref_emb)

    def close(k, scale=None):
        return max(0.0, 1 - abs(m[k] - prof[k]) / (scale or max(abs(prof[k]), 1e-6)))

    fullness = max(0.0, 1 - max(0.0, m["gaps"] - prof["gaps"]) * 4)
    composite = 100 * (
        0.45 * clap_sim + 0.20 * close("onset") + 0.15 * fullness
        + 0.10 * close("centroid") + 0.10 * close("width", 0.4)
    )
    return {"composite": round(composite, 1), "clap_sim": round(clap_sim, 3),
            **{k: round(v, 3) for k, v in m.items()}}
