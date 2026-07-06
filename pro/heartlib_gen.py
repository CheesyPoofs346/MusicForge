"""Pro engine CLI - Muse (Qwen backbone + MuCodec), called as a subprocess by
MusicForge's server (runs in this repo's isolated venv).

Two-step pipeline, per github.com/yuhui1038/Muse infer/README.md:
  1. batch_multi_generate.py         -> audio tokens (jsonl)
  2. MuCodec generate.py             -> tokens to wav

Prompt format verified 2026-07-05 against Muse/examples/muse_outputs/inputs.jsonl
(real training-data prompts): schema, [lyrics:]/[phoneme:] pairing, and
messages/role/content are all confirmed, not guesses.

Usage (same interface the server already calls):
  python heartlib_gen.py --tags-file T.txt --lyrics-file L.txt --seconds 120 --out out.mp3
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from muse_phonemes import phoneme_block

MUSE_DIR = Path(__file__).parent / "Muse"
MUCODEC_DIR = Path(__file__).parent / "MuCodec"
DEFAULT_CKPT = Path(__file__).parent / "ckpt"  # full HF checkpoint folder, not just model.safetensors

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _read(val: str, default: str = "") -> str:
    """--tags/--lyrics may be a raw string or a path to a file (server passes temp-file paths)."""
    if not val:
        return default
    p = Path(val)
    return p.read_text(encoding="utf-8").strip() if p.exists() else val


def _lyric_chunks(lyrics: str, count: int) -> list[str]:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return []
    size = max(1, (len(lines) + count - 1) // count)
    return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)]


# Keywords used to pull a section-specific clause out of the user's tags text
# (e.g. "...hook strips down to just kick..." -> routed to the Hook/Chorus
# sections) instead of repeating the entire tags blob identically on every
# section. Training prompts always describe each section differently; without
# this the model gets no signal for where a chorus/bridge/etc. should differ
# from the section before it - the main cause of "buggy"/directionless output,
# worse on instrumentals since there's no lyric text to differentiate sections either.
_SECTION_KEYWORDS = {
    "Intro": ["intro"], "Verse 1": ["verse"], "Verse 2": ["verse"],
    "Pre-Chorus": ["pre-chorus", "pre chorus"],
    "Chorus": ["chorus", "hook"], "Chorus 2": ["chorus", "hook"],
    "Hook": ["hook", "chorus"], "Final Hook": ["hook", "chorus"],
    "Extended Chorus": ["chorus", "hook"], "Final Chorus": ["chorus", "hook"],
    "Post-Chorus": ["chorus", "hook"], "Bridge": ["bridge"], "Bridge 2": ["bridge"],
    "Breakdown": ["breakdown"], "Interlude": ["interlude"], "Drop": ["drop"],
    "Solo": ["solo"], "Refrain": ["refrain"], "Outro": ["outro"], "Outro 2": ["outro"],
    "Coda": ["outro", "coda"], "Finale": ["outro", "finale"], "End": ["outro", "end"],
}


def _clause_for(tags: str, section_name: str) -> str:
    """The tags clause that names this section (e.g. 'hook strips down to just
    kick...'), if the user wrote one; else the whole tags blob unchanged."""
    for keyword in _SECTION_KEYWORDS.get(section_name, []):
        for clause in re.split(r',\s*', tags):
            if keyword in clause.lower():
                return clause.strip()
    return tags


def _build_prompt(tags: str, lyrics: str, seconds: int) -> dict:
    section_names = [
        "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Chorus 2", "Bridge",
        "Breakdown", "Final Chorus", "Outro", "Post-Chorus", "Solo", "Refrain",
        "Drop", "Hook", "Interlude", "Coda", "Finale", "Outro 2", "Tag",
        "Extended Chorus", "Bridge 2", "Final Hook", "End",
    ]
    section_count = max(4, min(len(section_names) + 1, round(seconds / 8)))
    intro = (
        "Please generate a song in the following style:"
        f"{tags}.\nNext, I will tell you the requirements and lyrics for the song fragment to be generated, "
        "section by section.\n"
        f"[Intro][desc:{tags}. Polished full-band production, clear musical structure, strong melody, "
        f"approximately {seconds} seconds.]"
    )
    messages = [{"role": "user", "content": intro}, {"role": "assistant", "content": ""}]
    if lyrics and lyrics.strip().lower() != "[instrumental]":
        chunks = _lyric_chunks(lyrics, section_count - 1)
        while len(chunks) < section_count - 1:
            chunks.append(chunks[-1] if chunks else lyrics)
        for name, chunk in zip(section_names, chunks):
            phonemes = phoneme_block(chunk.splitlines())
            desc = _clause_for(tags, name)
            messages += [
                {
                    "role": "user",
                    "content": (
                        f"[{name}][desc:{desc}. Expressive lead vocal with a complete arrangement.]"
                        f"[lyrics:\n{chunk}\n]{phonemes}"
                    ),
                },
                {"role": "assistant", "content": ""},
            ]
    else:
        for name in section_names[:section_count - 1]:
            desc = _clause_for(tags, name)
            messages += [
                {
                    "role": "user",
                    "content": f"[{name}][desc:{desc}. Instrumental development with clear transitions and full arrangement.]",
                },
                {"role": "assistant", "content": ""},
            ]
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--tags-file", default="")
    ap.add_argument("--lyrics", default="[instrumental]")
    ap.add_argument("--lyrics-file", default="")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    a = ap.parse_args()

    tags = _read(a.tags_file, a.tags)
    lyrics = _read(a.lyrics_file, a.lyrics)

    if not MUSE_DIR.exists():
        sys.exit(f"Muse repo not found at {MUSE_DIR} - clone github.com/yuhui1038/Muse there first")
    if not MUCODEC_DIR.exists():
        sys.exit(f"MuCodec repo not found at {MUCODEC_DIR} - clone github.com/tencent-ailab/MuCodec there first")
    if not Path(a.ckpt).exists():
        sys.exit(f"Muse checkpoint not found at {a.ckpt} - needs the FULL HF folder "
                  f"(config.json, tokenizer files, etc.), not just model.safetensors")
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_jsonl = td / "input.jsonl"
        tok_dir = td / "tokens"
        tok_dir.mkdir()
        in_jsonl.write_text(json.dumps(_build_prompt(tags, lyrics, a.seconds)) + "\n", encoding="utf-8")

        # --- step 1: audio tokens ---
        t0 = time.time()
        print("Muse: generating audio tokens...", flush=True)
        r = subprocess.run(
            [sys.executable, str(MUSE_DIR / "infer" / "batch_multi_generate.py"),
             "--input_path", str(in_jsonl),
             "--output_dir", str(tok_dir),
             "--ckpt_dir", a.ckpt,
             "--batch_size", "1",
             "--max_tokens", str(max(350, min(900, a.seconds * 12))),
             "--log_path", str(td / "error.log")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(MUSE_DIR), env=child_env,
        )
        if r.returncode != 0:
            sys.exit("Muse token generation failed: " + (r.stderr or r.stdout)[-800:])
        print(f"tokens generated in {time.time()-t0:.0f}s", flush=True)

        token_files = list(tok_dir.glob("*.jsonl"))
        if not token_files:
            sys.exit("Muse token generation produced no output - check " + str(td / "error.log"))
        generated = token_files[0].read_text(encoding="utf-8", errors="replace")
        if "<AUDIO_" not in generated:
            fail_path = Path(__file__).parent / "last_failed_tokens.jsonl"
            shutil.copy(token_files[0], fail_path)
            sys.exit("Muse generated no audio tokens; saved raw output to " + str(fail_path))

        # --- step 2: decode tokens to audio via MuCodec ---
        t0 = time.time()
        print("MuCodec: decoding tokens to audio...", flush=True)
        decode_out = td / "decoded"
        decode_out.mkdir()
        r = subprocess.run(
            [sys.executable, str(MUCODEC_DIR / "generate.py"),
             "--tokens", str(token_files[0]),
             "--out", str(decode_out / "origin.wav"),
             "--seconds", str(a.seconds)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(MUCODEC_DIR), env=child_env,
        )
        if r.returncode != 0:
            sys.exit("MuCodec decode failed: " + (r.stderr or r.stdout)[-800:])
        print(r.stdout, flush=True)  # ponytail: temp diagnostic, remove after stretch-ratio check
        print(f"decoded in {time.time()-t0:.0f}s", flush=True)

        wavs = list(decode_out.glob("*.wav"))
        if not wavs:
            sys.exit("MuCodec produced no wav")

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wavs[0], a.out)
        print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
