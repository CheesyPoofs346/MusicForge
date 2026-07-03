"""Turn a folder of audio into an ACE-Step training dataset.

Usage:
  1. Drop audio files (.mp3/.wav/.flac/.ogg) into training/raw/
  2. Optional: for any file, add a sidecar "<same name>.txt" with style tags
     (e.g. "synthwave, 80s, female vocals, dreamy"). Otherwise tags are
     auto-written by the local LLM from the filename.
  3. python prep.py              -> transcribes lyrics with Whisper
     python prep.py --instrumental  -> skips transcription, tags all as instrumental

Output: training/dataset/ (HuggingFace dataset the trainer reads).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for server.llm

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "dataset"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def transcribe(files):
    """Lyrics via faster-whisper; empty result -> [instrumental]."""
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cuda", compute_type="float16")
    out = {}
    for f in files:
        segments, _ = model.transcribe(str(f), vad_filter=True)
        text = "\n".join(s.text.strip() for s in segments).strip()
        out[f] = text if text else "[instrumental]"
        print(f"  {f.name}: {len(text)} chars of lyrics")
    return out


def auto_tags(files):
    """Sidecar .txt wins; otherwise the local LLM writes tags from the filename."""
    import server

    tags = {}
    need_llm = []
    for f in files:
        sidecar = f.with_suffix(".txt")
        if sidecar.exists():
            tags[f] = sidecar.read_text(encoding="utf-8").strip()
        else:
            need_llm.append(f)
    if need_llm:
        prompts = [
            "Based only on this music filename, guess plausible music style tags "
            "(genre, mood, instrumentation), comma-separated, max 15 words, "
            f"output only tags: {f.stem}"
            for f in need_llm
        ]
        for f, t in zip(need_llm, server.llm(prompts)):
            tags[f] = t.strip().strip('"') or f.stem.replace("_", ", ")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrumental", action="store_true", help="skip lyric transcription")
    args = ap.parse_args()

    files = sorted(f for f in RAW.iterdir() if f.suffix.lower() in AUDIO_EXTS) if RAW.exists() else []
    if not files:
        sys.exit(f"No audio in {RAW} — drop .mp3/.wav files there first.")
    print(f"{len(files)} audio files")

    tags = auto_tags(files)
    lyrics = {f: "[instrumental]" for f in files} if args.instrumental else transcribe(files)

    from datasets import Dataset

    rows = [
        {
            "keys": f.stem,
            "filename": str(f.resolve()),
            "tags": [t.strip() for t in tags[f].split(",") if t.strip()],
            "norm_lyrics": lyrics[f],
            "speaker_emb_path": "",
            "recaption": {},
        }
        for f in files
    ]
    Dataset.from_list(rows).save_to_disk(str(OUT))
    print(f"dataset written to {OUT} — now run: python train.py")


if __name__ == "__main__":
    main()
