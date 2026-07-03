"""Post-production mixing: split the generated track into stems (Demucs),
treat each like a mix engineer would, rebalance, and master.

Chain per stem (gentle on purpose — over-processing sounds worse than none):
  vocals : 90Hz high-pass, presence lift (2.5-5kHz), light compression, +1.5dB forward
  bass   : 35Hz high-pass (sub cleanup)
  drums  : untouched (Demucs drums are already punchy)
  other  : small 250-500Hz mud cut
Then sum -> 30Hz HPF -> -14 LUFS -> peak-safe.
"""
import threading
from pathlib import Path

import numpy as np

_model = None
_lock = threading.Lock()
STEM_SR = 44100


def _load():
    global _model
    with _lock:
        if _model is None:
            import torch
            from demucs.pretrained import get_model

            _model = get_model("htdemucs")
            _model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _model


def _split(data: np.ndarray, sr: int) -> dict:
    """(time, 2) float array -> {'vocals'|'drums'|'bass'|'other': (time, 2)} at 44.1kHz."""
    import julius
    import torch
    from demucs.apply import apply_model

    model = _load()
    wav = torch.from_numpy(data.T.astype(np.float32))
    if sr != STEM_SR:
        wav = julius.resample_frac(wav, sr, STEM_SR)
    ref = wav.mean(0)
    norm = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.inference_mode():
        sources = apply_model(model, norm[None].to(next(model.parameters()).device))[0].cpu()
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    return {name: sources[i].numpy().T for i, name in enumerate(model.sources)}


def _hpf(x, freq, sr):
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(2, freq, "highpass", fs=sr, output="sos"), x, axis=0)


def _band(x, lo, hi, sr):
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=sr, output="sos"), x, axis=0)


def _compress(x, sr, thresh_db=-18.0, ratio=3.0, release_ms=120.0):
    """Light feed-forward compressor; smoothed envelope via one-pole filter."""
    from scipy.signal import lfilter

    env = np.abs(x).max(axis=1) if x.ndim > 1 else np.abs(x)
    a = np.exp(-1.0 / (release_ms / 1000 * sr))
    env = lfilter([1 - a], [1, -a], env)
    env_db = 20 * np.log10(env + 1e-8)
    gain_db = np.minimum(0.0, (thresh_db - env_db) * (1 - 1 / ratio))
    gain = 10 ** (gain_db / 20)
    return x * (gain[:, None] if x.ndim > 1 else gain)


def db(x):
    return 10 ** (x / 20)


def _plate(x, sr, decay=0.6, wet=0.13):
    """Subtle plate-style reverb: dry vocals sound pasted-on; a short tail
    glues them into the track. Synthetic exponential-decay IR, low wet."""
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(7)
    n = int(sr * decay)
    env = np.exp(-6.0 * np.arange(n) / n)
    ir = np.stack([rng.standard_normal(n) * env, rng.standard_normal(n) * env], 1)
    w = np.stack([fftconvolve(x[:, i], ir[:, i])[: len(x)] for i in (0, 1)], 1)
    w *= (np.abs(x).max() + 1e-9) / (np.abs(w).max() + 1e-9)  # match level before blending
    return x * (1 - wet) + w * wet


def _enhance_vocals(v, sr):
    """Neural restoration of the vocal stem (resemble-enhance): removes the
    synthetic fizz and restores presence. The single biggest anti-'AI-voice' lever."""
    import os
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath  # its checkpoint stores posix paths
    import torch
    from resemble_enhance.enhancer.inference import enhance

    wav = torch.from_numpy(v.mean(axis=1).astype(np.float32))
    out, out_sr = enhance(
        wav, sr, "cuda" if torch.cuda.is_available() else "cpu",
        nfe=32, solver="midpoint", lambd=0.9, tau=0.5,
    )
    out = out.cpu().numpy()
    if out_sr != sr:
        import julius

        out = julius.resample_frac(torch.from_numpy(out), out_sr, sr).numpy()
    out = np.stack([out, out], 1)[: len(v)]
    if len(out) < len(v):
        out = np.pad(out, ((0, len(v) - len(out)), (0, 0)))
    out *= (np.abs(v).max() + 1e-9) / (np.abs(out).max() + 1e-9)  # level-match
    return out


def mix_track(path: Path):
    """Stem-split, treat, rebalance, and write back (44.1kHz stereo)."""
    import os

    import soundfile as sf

    data, sr = sf.read(path)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    stems = _split(data, sr)
    sr = STEM_SR

    voc = stems["vocals"]
    if os.environ.get("VOCAL_ENHANCE", "on") != "off":
        try:
            voc = _enhance_vocals(voc, sr)
        except Exception:
            pass  # ponytail: enhancement is polish — never fail the track over it
    v = _hpf(voc, 90, sr)
    v = v + 0.25 * _band(v, 2500, 5000, sr)        # presence: vocals cut through
    v = _compress(v, sr)                            # steady level
    v = _plate(v, sr) * db(1.5)                     # glued into the track, sit forward
    b = _hpf(stems["bass"], 35, sr)
    o = stems["other"] - 0.15 * _band(stems["other"], 250, 500, sr)  # mud cut
    mixed = v + stems["drums"] + b + o

    sf.write(path, mixed, sr)  # final polish (30Hz HPF, LUFS, peak) = server.master_audio
