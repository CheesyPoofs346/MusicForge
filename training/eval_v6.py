"""Score v6-diction checkpoints on musicality (composite) AND intelligibility (WER),
against the v5+enhance baseline (76.5 composite / 0.557 WER on the same tests)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/musicforge")
import eval as ev
import sweep
import sweep_intel

HERE = Path(__file__).parent
OUT = HERE / "sweep_out"
v6dir = sorted((HERE / "exps" / "logs" / "lightning_logs").glob("*musicforge_v6_diction"))[-1] / "checkpoints"

results = {}
for d in sorted(p for p in v6dir.iterdir() if p.name.endswith("_lora")):
    steps = d.name.split("step=")[1].split("_")[0]
    cname = f"v6d-{steps}"
    results[cname] = {}
    for genre, prompt, lyrics, seed in sweep.TESTS[:2]:
        f = OUT / f"{cname}_{genre}.wav"
        if not f.exists():
            sweep.gen(prompt, lyrics, str(d), 20.0, seed, f, 1.0)
        comp = ev.score(f, genre)["composite"]
        w = sweep_intel.wer_of(f, lyrics)
        results[cname][genre] = {"composite": comp, "wer": w}
        print(f"{cname:>9} {genre:>8}: composite {comp}  WER {w}", flush=True)

print("\nbaseline v5+enhance: 76.5 / WER 0.557")
for c, gs in results.items():
    mc = sum(v["composite"] for v in gs.values()) / len(gs)
    mw = sum(v["wer"] for v in gs.values()) / len(gs)
    print(f"{c:>9}: {mc:.1f} / WER {mw:.3f}")
(HERE / "v6_results.json").write_text(json.dumps(results, indent=1))
