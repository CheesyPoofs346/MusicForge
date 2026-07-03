"""Config sweep: generate fixed-seed test tracks per config, score against chart
profiles, print a ranked table. Fixed prompts + fixed lyrics + fixed seeds so the
only variable is the config."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/musicforge")
import eval as ev
import mix
import server

HERE = Path(__file__).parent
OUT = HERE / "sweep_out"
OUT.mkdir(exist_ok=True)
LOGS = HERE / "exps" / "logs" / "lightning_logs"

V1 = str(LOGS / "2026-07-01_15-26-12musicforge_v1" / "checkpoints" / "epoch=5-step=1500_lora")
V2 = str(LOGS / "2026-07-01_16-54-19musicforge_v2" / "checkpoints")

LYRICS = """[verse]
Started with a whisper now they hear me for miles
Every single setback turned to fuel in the fire
[chorus]
Louder, louder, turn it up till it breaks (up till it breaks)
We don't stop, we don't sleep, we escape (we escape)
"""

TESTS = [
    ("hip-hop", "aggressive rap anthem at 92 BPM with pounding 808s and crisp hats; verse rides tight double-time flow, pre-chorus strips to claps and sub pulse, chorus slams with gang vocals and brass stabs, glitch fills between lines, loud punchy club mix", LYRICS, 41),
    ("pop", "anthemic pop banger at 118 BPM with driving live drums and shimmering synth layers; verse rides plucky bass groove, pre-chorus lifts with stacked harmonies, chorus explodes with wall-of-sound guitars and claps, bright radio-loud mix", LYRICS, 42),
    ("dance", "festival house track at 126 BPM with four-on-the-floor kick and rolling bassline; intro builds with filtered risers, drop slams with big room lead and crowd shouts, breakdown strips to piano chords then rebuilds, wide loud club mix", "[instrumental]", 43),
]

CONFIGS = [
    ("base", "none", 15.0, 1.0),
    ("v1-1500", V1, 15.0, 1.0),
    ("v2-1000", f"{V2}\\epoch=1-step=1000_lora", 15.0, 1.0),
    ("v2-2000", f"{V2}\\epoch=2-step=2000_lora", 15.0, 1.0),
    ("v2-3000", f"{V2}\\epoch=3-step=3000_lora", 15.0, 1.0),
]

# stage 3: the two stage-2 leaders combined
CONFIGS3 = [
    ("v1-g20", V1, 20.0, 1.0),
]

# stage 5: vocal-quality knobs on the production model (v3rap-1000 + g20)
V3R = None  # filled in __main__ from CONFIGS4
CONFIGS5_EXTRAS = {
    "heun": {"scheduler_type": "heun"},
    "erg": {"guidance_scale_text": 5.0, "guidance_scale_lyric": 1.5},
    "heun-erg": {"scheduler_type": "heun", "guidance_scale_text": 5.0, "guidance_scale_lyric": 1.5},
}

# stage 4: genre-focused v3-rap checkpoints, at the winning guidance
V3 = str(LOGS / "2026-07-01_20-33-18musicforge_v3_rap" / "checkpoints")
CONFIGS4 = [
    ("v3rap-1000-g20", f"{V3}\\epoch=6-step=1000_lora", 20.0, 1.0),
    ("v3rap-2000-g20", f"{V3}\\epoch=13-step=2000_lora", 20.0, 1.0),
]

# stage 2: LoRA-weight and guidance variants around the stage-1 leaders
CONFIGS2 = [
    ("v1-w05-g15", V1, 15.0, 0.5),
    ("v1-w05-g10", V1, 10.0, 0.5),
    ("v1-w10-g10", V1, 10.0, 1.0),
    ("base-g10", "none", 10.0, 1.0),
    ("base-g20", "none", 20.0, 1.0),
]


def gen(prompt, lyrics, lora, guidance, seed, out_path, weight=1.0, **extra):
    pipe = server.load_ace()
    pipe(
        format="wav", audio_duration=60.0, prompt=prompt, lyrics=lyrics,
        infer_step=120, guidance_scale=guidance, lora_name_or_path=lora,
        lora_weight=weight, manual_seeds=[seed], save_path=str(out_path), **extra,
    )
    if lora == "none":
        pipe.lora_path = "none"  # upstream forgets to reset after unload
    out_path.with_name(out_path.stem + "_input_params.json").unlink(missing_ok=True)
    mix.mix_track(out_path)
    server.master_audio(out_path)


def run(configs, results_file, tests=TESTS):
    results = {}
    if Path(results_file).exists():
        results = json.loads(Path(results_file).read_text())
    for cname, lora, guidance, weight, *rest in configs:
        extra = rest[0] if rest else {}
        results.setdefault(cname, {})
        for genre, prompt, lyrics, seed in tests:
            if genre in results[cname]:
                continue
            f = OUT / f"{cname}_{genre}.wav"
            t = time.time()
            gen(prompt, lyrics, lora, guidance, seed, f, weight, **extra)
            s = ev.score(f, genre)
            s["gen_s"] = round(time.time() - t, 1)
            results[cname][genre] = s
            print(f"{cname:>10} {genre:>8}: composite {s['composite']}  clap {s['clap_sim']}  gaps {s['gaps']}  onset {s['onset']}", flush=True)
            Path(results_file).write_text(json.dumps(results, indent=1))
    print("\n=== RANKING (mean composite) ===")
    for cname, gs in sorted(results.items(), key=lambda kv: -sum(v["composite"] for v in kv[1].values()) / len(kv[1])):
        mean = sum(v["composite"] for v in gs.values()) / len(gs)
        print(f"{cname:>10}: {mean:.1f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    if stage == "7":
        v5dir = sorted(LOGS.glob("*musicforge_v5_band"))[-1] / "checkpoints"
        cfgs = sorted(
            (f"v5b-{d.name.split('step=')[1].split('_')[0]}", str(d), 20.0, 1.0)
            for d in v5dir.iterdir() if d.is_dir() and d.name.endswith("_lora")
        )
        run(cfgs, HERE / "sweep_results.json", tests=TESTS[:2])
    elif stage == "6":
        v4dir = next(LOGS.glob("*musicforge_v4_vocal")) / "checkpoints"
        cfgs = sorted(
            (f"v4v-{d.name.split('step=')[1].split('_')[0]}", str(d), 20.0, 1.0)
            for d in v4dir.iterdir() if d.is_dir() and d.name.endswith("_lora")
        )
        run(cfgs, HERE / "sweep_results.json", tests=TESTS[:2])  # vocal tests only
    elif stage == "5":
        v3r = f"{V3}\\epoch=6-step=1000_lora"
        cfgs = [(f"v3r-{n}", v3r, 20.0, 1.0, ex) for n, ex in CONFIGS5_EXTRAS.items()]
        run(cfgs, HERE / "sweep_results.json", tests=TESTS[:2])  # vocal tests only
    elif stage == "4":
        run(CONFIGS4, HERE / "sweep_results.json")
    elif stage == "3":
        run(CONFIGS3, HERE / "sweep_results.json")
    elif stage == "2":
        run(CONFIGS2, HERE / "sweep_results.json")  # same file: one global ranking
    else:
        run(CONFIGS, HERE / "sweep_results.json")
