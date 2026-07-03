"""v7 verdict: composite (chart similarity) + WER (word clarity) + singing
phrasing (note stats vs the chart envelope), per checkpoint, vs the v5 champion.

Champion baselines on the same fixed tests (v5b-1500 + enhance = intel_lg0):
  composite 76.5 · WER 0.557
  pop phrasing: sustain 0.54, longest 0.92s, 31 notes  (chart pop: ~0.3 / ~1s / ~17)
  hip-hop phrasing: singing 46.4, sustain 0.09 (talks)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/musicforge")
import eval as ev
import singing
import sweep
import sweep_intel

HERE = Path(__file__).parent
OUT = HERE / "sweep_out"
v7dir = sorted((HERE / "exps" / "logs" / "lightning_logs").glob("*musicforge_v7_phrasing"))[-1] / "checkpoints"

results = {}
for d in sorted(p for p in v7dir.iterdir() if p.name.endswith("_lora")):
    steps = d.name.split("step=")[1].split("_")[0]
    cname = f"v7p-{steps}"
    results[cname] = {}
    for genre, prompt, lyrics, seed in sweep.TESTS[:2]:
        f = OUT / f"{cname}_{genre}.wav"
        if not f.exists():
            sweep.gen(prompt, lyrics, str(d), 20.0, seed, f, 1.0)
        comp = ev.score(f, genre)["composite"]
        w = sweep_intel.wer_of(f, lyrics)
        s = singing.singing_metrics(f)
        results[cname][genre] = {"composite": comp, "wer": w, **s}
        print(f"{cname:>9} {genre:>8}: comp {comp}  WER {w}  sing {s['singing']}  "
              f"sustain {s['sustain_frac']}  longest {s['longest_note_s']}s  notes {s['notes']}", flush=True)

(HERE / "v7_results.json").write_text(json.dumps(results, indent=1))
print("\nchampion: comp 76.5 / WER 0.557 / pop longest 0.92s / hip-hop sing 46.4")
