"""Calibrate the singing detector and baseline every era of the model.
Chart pop/R&B refs MUST score high (they're sung) or the detector is wrong;
chart hip-hop should land mid; our talking-not-singing outputs should be low."""
import json
from pathlib import Path

import singing

HERE = Path(__file__).parent
OUT = HERE / "sweep_out"
REFS = HERE / "refs"

rows = {}
for genre in ("pop", "rnb", "hip-hop"):
    for f in sorted((REFS / genre).glob("*.mp3"))[:4]:
        rows[f"REF-{genre}-{f.stem}"] = singing.singing_metrics(f)

for name in ("base_pop", "base_hip-hop", "v4v-1000_pop", "v4v-1000_hip-hop",
             "v5b-1500_pop", "v5b-1500_hip-hop", "intel_lg0_pop", "intel_lg0_hip-hop"):
    f = OUT / f"{name}.wav"
    if f.exists():
        rows[name] = singing.singing_metrics(f)

for k, m in rows.items():
    print(f"{k:>28}: singing {m['singing']:>5}  sustain {m['sustain_frac']:.2f}  "
          f"range {m['range_st']:>4}st  notes {m['notes']}  longest {m['longest_note_s']}s", flush=True)
(HERE / "singing_baseline.json").write_text(json.dumps(rows, indent=1))
