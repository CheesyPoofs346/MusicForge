"""Vocal intelligibility sweep: does turning up ACE's lyric guidance make the
words audible? Measures Word-Error-Rate (Whisper transcript of the generated
vocal stem vs the intended lyrics) alongside the usual chart-composite, so we
can buy clarity without paying in musicality."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "C:/musicforge")
import eval as ev
import mix
import sweep

HERE = Path(__file__).parent
OUT = HERE / "sweep_out"

CONFIGS = [
    ("lg0", {}),                                             # current production
    ("lg15", {"guidance_scale_lyric": 1.5}),
    ("lg3", {"guidance_scale_lyric": 3.0}),
    ("lg15-tg2", {"guidance_scale_lyric": 1.5, "guidance_scale_text": 2.0}),
]

_whisper = None


def wer_of(path, intended_lyrics):
    global _whisper
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from jiwer import wer

    if _whisper is None:
        _whisper = WhisperModel("small", device="cuda", compute_type="float16")
    data, sr = sf.read(path)
    if data.ndim == 1:
        data = np.stack([data, data], 1)
    voc = mix._split(data, sr)["vocals"].mean(axis=1).astype("float32")
    # whisper assumes 16kHz for raw arrays — resample or it hears chipmunks
    import librosa

    voc = librosa.resample(voc, orig_sr=mix.STEM_SR, target_sr=16000)
    segments, _ = _whisper.transcribe(voc, language="en", vad_filter=True)
    heard = " ".join(s.text for s in segments)
    clean = lambda t: re.sub(r"[^a-z' ]+", " ", re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t.lower())).split()
    ref, hyp = " ".join(clean(intended_lyrics)), " ".join(clean(heard))
    return round(wer(ref, hyp), 3) if ref else 1.0


V5 = str(sorted((HERE / "exps" / "logs" / "lightning_logs").glob("*musicforge_v5_band"))[-1]
         / "checkpoints" / "epoch=8-step=1500_lora")


def main():
    results = {}
    for cname, extra in CONFIGS:
        results[cname] = {}
        for genre, prompt, lyrics, seed in sweep.TESTS[:2]:
            f = OUT / f"intel_{cname}_{genre}.wav"
            if not f.exists():
                sweep.gen(prompt, lyrics, V5, 20.0, seed, f, 1.0, **extra)
            comp = ev.score(f, genre)["composite"]
            w = wer_of(f, lyrics)
            results[cname][genre] = {"composite": comp, "wer": w}
            print(f"{cname:>9} {genre:>8}: composite {comp}  WER {w}", flush=True)
    (HERE / "intel_results.json").write_text(json.dumps(results, indent=1))
    print("\n=== mean composite / mean WER (lower WER = clearer words) ===")
    for c, gs in results.items():
        mc = sum(v["composite"] for v in gs.values()) / len(gs)
        mw = sum(v["wer"] for v in gs.values()) / len(gs)
        print(f"{c:>9}: {mc:.1f} / WER {mw:.3f}")


if __name__ == "__main__":
    main()
