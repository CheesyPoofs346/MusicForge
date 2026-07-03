# MusicForge v7-1000

LoRA fine-tune of [ACE-Step](https://github.com/ace-step/ACE-Step) (`ACE-Step/ACE-Step-v1-3.5B`), trained locally — v7 in the lineage documented in [`training/tuning_report.md`](../training/tuning_report.md).

**Trained on:** 84 Creative-Commons tracks (FMA), curated for melodic vocal phrasing — pitch-tracked to match the sustain/range/note-length envelope of real sung vocals, singing genres only (pop, R&B, folk, country, rock). 1000 steps.

**Measured result** (see tuning report for methodology): fixes the "talking, not singing" failure mode of earlier checkpoints — hip-hop test track went from a singing-detector score of 46.4 (spoken cadence) to 95.8 (held notes, chart-realistic phrasing), while composite chart-similarity hit a project-record 85.7 on the pop test, with word-clarity (WER) held flat vs the previous champion.

## Use it

```
set ACE_LORA=C:\path\to\musicforge-v7-1000.safetensors
```
Or point `lora_active.txt` at a folder containing this file renamed to `pytorch_lora_weights.safetensors` — that's the layout `server.py` expects (see `training/README.md`).
